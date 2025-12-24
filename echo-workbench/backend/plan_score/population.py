from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pydicom
import SimpleITK as sitk

from .metrics import compute_metric_from_dose, required_structures
from .objectives import objectives_for_protocol
from .score import compute_population_scores


def _load_ct_meta(case_dir: Path) -> dict[str, Any]:
    meta_path = case_dir / "CT_MetaData.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _reference_image(meta: dict[str, Any], mask_shape: tuple[int, int, int] | None = None) -> sitk.Image:
    if mask_shape is not None:
        size = (int(mask_shape[2]), int(mask_shape[1]), int(mask_shape[0]))
    else:
        size = tuple(int(x) for x in meta.get("size_xyz_mm", []))
    spacing = tuple(float(x) for x in meta.get("resolution_xyz_mm", []))
    origin = tuple(float(x) for x in meta.get("origin_xyz_mm", []))
    direction = tuple(float(x) for x in meta.get("direction", []))
    if len(size) != 3 or len(spacing) != 3 or len(origin) != 3 or len(direction) != 9:
        raise ValueError("CT metadata missing size/spacing/origin/direction")
    img = sitk.Image(size, sitk.sitkFloat32)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    return img


def _load_dose(case_dir: Path, ref_image: sitk.Image) -> np.ndarray:
    dose_path = case_dir / "DicomFiles" / "rt_dose_echo_imrt.dcm"
    if not dose_path.exists():
        raise FileNotFoundError(f"RTDOSE not found: {dose_path}")
    dose_img = sitk.ReadImage(str(dose_path))
    dose_img = sitk.Cast(dose_img, sitk.sitkFloat32)
    resampled = sitk.Resample(
        dose_img,
        ref_image,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    dose_array = sitk.GetArrayFromImage(resampled)
    dose_meta = pydicom.dcmread(str(dose_path), stop_before_pixels=True)
    scale = float(getattr(dose_meta, "DoseGridScaling", 1.0))
    return dose_array * scale


def _load_masks(case_dir: Path, structures: set[str]) -> dict[str, np.ndarray]:
    mask_path = case_dir / "StructureSet_Data.h5"
    masks = {}
    with h5py.File(mask_path, "r") as handle:
        for name in structures:
            if name in ("LUNGS_NOT_GTV",):
                continue
            if name in handle:
                masks[name] = handle[name][...].astype(bool)
    if "LUNGS_NOT_GTV" in structures:
        lung_l = masks.get("LUNG_L")
        lung_r = masks.get("LUNG_R")
        gtv = masks.get("GTV")
        if lung_l is not None or lung_r is not None:
            lung_union = None
            if lung_l is not None and lung_r is not None:
                lung_union = lung_l | lung_r
            else:
                lung_union = lung_l if lung_l is not None else lung_r
            if lung_union is not None:
                if gtv is not None:
                    masks["LUNGS_NOT_GTV"] = lung_union & ~gtv
                else:
                    masks["LUNGS_NOT_GTV"] = lung_union
    return masks


def build_population_metrics(
    protocol_name: str,
    data_root: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    objectives = objectives_for_protocol(protocol_name)
    structures = required_structures(objectives)

    patients: list[dict[str, Any]] = []
    case_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("Lung_Patient_")])
    if limit:
        case_dirs = case_dirs[:limit]

    for case_dir in case_dirs:
        meta = _load_ct_meta(case_dir)
        masks = _load_masks(case_dir, structures)
        mask_shape = None
        for mask in masks.values():
            mask_shape = mask.shape
            break
        ref_img = _reference_image(meta, mask_shape)
        dose_gy = _load_dose(case_dir, ref_img)
        if mask_shape and dose_gy.shape != mask_shape:
            min_z = min(dose_gy.shape[0], mask_shape[0])
            min_y = min(dose_gy.shape[1], mask_shape[1])
            min_x = min(dose_gy.shape[2], mask_shape[2])
            dose_gy = dose_gy[:min_z, :min_y, :min_x]
            for key, mask in list(masks.items()):
                masks[key] = mask[:min_z, :min_y, :min_x]
        metrics = {}
        for obj in objectives:
            metrics[obj.id] = compute_metric_from_dose(obj, dose_gy, masks)
        patients.append({"case_id": case_dir.name, "metrics": metrics})

    population = {
        "protocol": protocol_name,
        "patients": patients,
        "objectives": [obj.__dict__ for obj in objectives],
    }
    population["population_scores"] = compute_population_scores(population, objectives)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "population_metrics.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(population, handle, indent=2)
    return population
