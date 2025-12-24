import asyncio
import mimetypes
import io
import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

import h5py
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from backend.runner import _default_data_dir, run_echo_example, run_compressrtp, _ensure_echo_vmat_on_path
except ImportError:
    from runner import _default_data_dir, run_echo_example, run_compressrtp, _ensure_echo_vmat_on_path

try:
    from adapters.example_adapter import ExampleAdapter
    from adapters.huggingface_adapter import HuggingFaceAdapter
except ImportError:
    from backend.adapters.example_adapter import ExampleAdapter
    from backend.adapters.huggingface_adapter import HuggingFaceAdapter


RUNS_DIR = Path(__file__).resolve().parent / "runs"
RUNS_DIR_COMPRESS = Path(__file__).resolve().parent / "runs-compressrtp"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR_COMPRESS.mkdir(parents=True, exist_ok=True)
BATCH_DIR = Path(__file__).resolve().parent / "runs-batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
WORKBENCH_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKBENCH_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DICOM_DIR = PROCESSED_DIR / "dicom"
DICOM_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURE_COLORS = [
    (96, 165, 250),
    (249, 115, 22),
    (34, 211, 238),
    (250, 204, 21),
    (52, 211, 153),
    (244, 114, 182),
    (167, 139, 250),
    (248, 113, 113),
    (56, 189, 248),
    (251, 113, 133),
    (74, 222, 128),
    (232, 121, 249),
]


class RunRequest(BaseModel):
    case_id: str
    protocol: str = "Lung_2Gy_30Fx"
    adapter: Literal["example", "huggingface"] = "example"
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    hf_subdir: Optional[str] = None
    optimizer: Literal["echo-vmat", "compressrtp"] = "echo-vmat"
    use_planner_beams: bool = False
    use_available_beams: bool = False
    force_sparse: bool = False
    fast: bool = False
    super_fast: bool = False
    compress_mode: Literal["sparse-only", "sparse-plus-low-rank", "wavelet"] = "sparse-only"
    threshold_perc: float = 10.0
    rank: int = 5
    step: Optional[Literal["ddc", "sparse", "svd", "wavelet"]] = None
    beam_ids: Optional[list[int]] = None
    use_gpu: bool = False
    tag: Optional[str] = None


class RunResponse(BaseModel):
    run_id: str


class BatchRequest(BaseModel):
    runs: list[RunRequest]
    label: Optional[str] = None


class BatchResponse(BaseModel):
    batch_id: str
    run_ids: list[str]


app = FastAPI()
access_logger = logging.getLogger("uvicorn.access")
access_logger.disabled = True
access_logger.propagate = False
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_file(path: Path, chunk_size: int = 1024 * 1024):
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


def _run_roots() -> dict[str, Path]:
    return {"echo-vmat": RUNS_DIR, "compressrtp": RUNS_DIR_COMPRESS}


def _batch_path(batch_id: str) -> Path:
    return BATCH_DIR / f"{batch_id}.json"


def _safe_run_id(case_id: str, optimizer: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_case = re.sub(r"[^A-Za-z0-9_-]+", "_", case_id or "case").strip("_")
    suffix = uuid4().hex[:6]
    return f"{optimizer}-{safe_case}-{ts}-{suffix}"


def _write_batch_status(batch_id: str, payload: dict) -> None:
    _write_json(_batch_path(batch_id), payload)


def _read_batch_status(batch_id: str) -> dict:
    path = _batch_path(batch_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="batch not found")
    return _read_json(path)


def _find_run_dir(run_id: str) -> Path:
    for root in _run_roots().values():
        candidate = root / run_id
        if candidate.exists():
            return candidate
    raise HTTPException(status_code=404, detail="run not found")


def _load_run_config(run_id: str) -> dict:
    out_dir = _find_run_dir(run_id)
    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="config.json not found for run")
    return _read_json(config_path)


def _resolve_case_dir(run_id: str) -> tuple[Path, dict]:
    config = _load_run_config(run_id)
    case_id = config.get("case_id")
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id missing in run config")
    data_dir = Path(config.get("data_dir") or _default_data_dir())
    direct = data_dir / case_id
    nested = data_dir / case_id / case_id
    if direct.is_dir():
        return direct, config
    if nested.is_dir():
        return nested, config
    raise HTTPException(status_code=404, detail="case data directory not found")


def _ct_reference(case_dir: Path) -> tuple[dict, Path, str]:
    meta_path = case_dir / "CT_MetaData.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="CT_MetaData.json not found")
    meta = _read_json(meta_path)
    file_ref = meta.get("ct_hu_3d_File", "CT_Data.h5/ct_hu_3d")
    if "/" not in file_ref:
        raise HTTPException(status_code=500, detail="Invalid ct_hu_3d_File reference")
    file_name, dataset = file_ref.split("/", 1)
    ct_path = case_dir / file_name
    if not ct_path.exists():
        raise HTTPException(status_code=404, detail="CT data file not found")
    return meta, ct_path, dataset


def _structure_entries(case_dir: Path) -> list[dict]:
    meta_path = case_dir / "StructureSet_MetaData.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="StructureSet_MetaData.json not found")
    meta = _read_json(meta_path)
    entries = []
    for item in meta:
        name = item.get("name")
        file_ref = item.get("structure_mask_3d_File")
        if not name or not file_ref or "/" not in file_ref:
            continue
        file_name, dataset = file_ref.split("/", 1)
        path = case_dir / file_name
        if not path.exists():
            continue
        entries.append({"name": name, "path": path, "dataset": dataset})
    if not entries:
        raise HTTPException(status_code=404, detail="No structure masks found")
    return entries


def _rt_plan_template(case_dir: Path) -> Path:
    candidates = [
        case_dir / "rt_plan_echo_vmat.dcm",
        case_dir / "rt_plan_echo_imrt.dcm",
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in sorted(case_dir.glob("rt_plan*.dcm")):
        return path
    raise HTTPException(status_code=404, detail="RT plan template DICOM not found")


def _rt_dose_template(case_dir: Path) -> Path:
    candidates = [
        case_dir / "rt_dose_echo_vmat.dcm",
        case_dir / "rt_dose_echo_imrt.dcm",
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in sorted(case_dir.glob("rt_dose*.dcm")):
        return path
    raise HTTPException(status_code=404, detail="RT dose template DICOM not found")


def _dicom_reference(case_dir: Path) -> dict:
    try:
        from pydicom import dcmread
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"pydicom not available: {exc}") from exc
    template_path = _rt_plan_template(case_dir)
    ds = dcmread(str(template_path))
    return {
        "study_uid": getattr(ds, "StudyInstanceUID", None),
        "frame_uid": getattr(ds, "FrameOfReferenceUID", None),
        "patient_name": str(getattr(ds, "PatientName", "")) if hasattr(ds, "PatientName") else None,
        "patient_id": getattr(ds, "PatientID", None),
        "study_date": getattr(ds, "StudyDate", None),
        "study_time": getattr(ds, "StudyTime", None),
        "study_description": getattr(ds, "StudyDescription", None),
    }


def _ct_dicom_dir(case_id: str) -> Path:
    return DICOM_DIR / case_id / "ct"


def _rt_struct_dir(case_id: str) -> Path:
    return DICOM_DIR / case_id / "rtstruct"


def _update_dicom_uids(ds, series_uid: str, sop_uid: str) -> None:
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = sop_uid
    if hasattr(ds, "file_meta"):
        ds.file_meta.MediaStorageSOPInstanceUID = sop_uid


def _write_ct_dicom(case_dir: Path, case_id: str, overwrite: bool = False) -> dict:
    ct_dir = _ct_dicom_dir(case_id)
    if ct_dir.exists() and not overwrite:
        existing = sorted(ct_dir.glob("*.dcm"))
        if existing:
            return {"status": "exists", "ct_dir": str(ct_dir), "slices": len(existing)}
    ct_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in ct_dir.glob("*.dcm"):
            path.unlink()
    meta, ct_path, dataset = _ct_reference(case_dir)
    try:
        from datetime import datetime
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"pydicom not available: {exc}") from exc
    ref = _dicom_reference(case_dir)
    study_uid = ref.get("study_uid") or generate_uid()
    frame_uid = ref.get("frame_uid") or generate_uid()
    patient_name = ref.get("patient_name") or case_id
    patient_id = ref.get("patient_id") or case_id
    study_date = ref.get("study_date") or datetime.utcnow().strftime("%Y%m%d")
    study_time = ref.get("study_time") or datetime.utcnow().strftime("%H%M%S")
    series_uid = generate_uid()
    origin = meta.get("origin_xyz_mm", [0.0, 0.0, 0.0])
    spacing = meta.get("resolution_xyz_mm", [1.0, 1.0, 1.0])
    direction = meta.get("direction") or [1, 0, 0, 0, 1, 0, 0, 0, 1]
    orientation = [
        float(direction[0]),
        float(direction[1]),
        float(direction[2]),
        float(direction[3]),
        float(direction[4]),
        float(direction[5]),
    ]
    with h5py.File(ct_path, "r") as handle:
        ct_data = handle[dataset]
        z_slices, rows, cols = ct_data.shape
        for z in range(z_slices):
            slice_arr = np.asarray(ct_data[z, :, :], dtype=np.int16)
            sop_uid = generate_uid()
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = CTImageStorage
            file_meta.MediaStorageSOPInstanceUID = sop_uid
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = generate_uid()
            file_path = ct_dir / f"ct_{z + 1:04d}.dcm"
            ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
            ds.is_little_endian = True
            ds.is_implicit_VR = False
            ds.SOPClassUID = CTImageStorage
            ds.SOPInstanceUID = sop_uid
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.FrameOfReferenceUID = frame_uid
            ds.PatientName = patient_name
            ds.PatientID = patient_id
            ds.StudyDate = study_date
            ds.StudyTime = study_time
            ds.Modality = "CT"
            ds.SeriesNumber = 1
            ds.InstanceNumber = z + 1
            ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
            ds.ImageOrientationPatient = orientation
            ds.ImagePositionPatient = [
                float(origin[0]),
                float(origin[1]),
                float(origin[2] + z * spacing[2]),
            ]
            ds.SliceLocation = float(origin[2] + z * spacing[2])
            ds.SliceThickness = float(spacing[2])
            ds.SpacingBetweenSlices = float(spacing[2])
            ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
            ds.Rows = int(rows)
            ds.Columns = int(cols)
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1
            ds.RescaleIntercept = 0
            ds.RescaleSlope = 1
            ds.RescaleType = "HU"
            ds.PixelData = np.ascontiguousarray(slice_arr).tobytes()
            ds.save_as(str(file_path), write_like_original=False)
    _write_json(
        ct_dir / "ct_series.json",
        {
            "case_id": case_id,
            "study_instance_uid": study_uid,
            "series_instance_uid": series_uid,
            "frame_of_reference_uid": frame_uid,
            "origin_xyz_mm": origin,
            "resolution_xyz_mm": spacing,
            "shape_zyx": [int(z_slices), int(rows), int(cols)],
        },
    )
    return {"status": "created", "ct_dir": str(ct_dir), "slices": int(z_slices)}


def _write_rt_struct_dicom(
    case_dir: Path,
    case_id: str,
    overwrite: bool = False,
) -> dict:
    rt_dir = _rt_struct_dir(case_id)
    rt_path = rt_dir / "rt_struct_portpy.dcm"
    if rt_path.exists() and not overwrite:
        return {"status": "exists", "artifact": str(rt_path)}
    rt_dir.mkdir(parents=True, exist_ok=True)
    if overwrite and rt_path.exists():
        rt_path.unlink()
    ct_result = _write_ct_dicom(case_dir, case_id, overwrite=False)
    ct_dir = _ct_dicom_dir(case_id)
    ct_files = sorted(ct_dir.glob("*.dcm"))
    if not ct_files:
        raise HTTPException(status_code=404, detail="CT DICOM series not found")
    try:
        from datetime import datetime
        from pydicom import dcmread
        from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
        from pydicom.sequence import Sequence
        from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, RTStructureSetStorage, generate_uid
        from skimage import measure
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"missing dependency: {exc}") from exc
    ct_first = None
    ct_sops = []
    ct_positions = []
    for path in ct_files:
        ds = dcmread(str(path), stop_before_pixels=True)
        if ct_first is None:
            ct_first = ds
        ct_sops.append(ds.SOPInstanceUID)
        ct_positions.append(ds.ImagePositionPatient)
    if ct_first is None:
        raise HTTPException(status_code=404, detail="CT DICOM series not found")
    meta, _ct_path, _dataset = _ct_reference(case_dir)
    orientation = ct_first.ImageOrientationPatient
    col_dir = np.array(orientation[0:3], dtype=float)
    row_dir = np.array(orientation[3:6], dtype=float)
    row_spacing, col_spacing = ct_first.PixelSpacing
    series_uid = generate_uid()
    sop_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = RTStructureSetStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(rt_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = RTStructureSetStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = ct_first.StudyInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = ct_first.FrameOfReferenceUID
    ds.PatientName = ct_first.PatientName
    ds.PatientID = ct_first.PatientID
    ds.StudyDate = ct_first.StudyDate
    ds.StudyTime = ct_first.StudyTime
    ds.Modality = "RTSTRUCT"
    ds.StructureSetLabel = f"{case_id}_PortPy"
    ds.StructureSetDate = ct_first.StudyDate or datetime.utcnow().strftime("%Y%m%d")
    ds.StructureSetTime = ct_first.StudyTime or datetime.utcnow().strftime("%H%M%S")
    ref_frame = Dataset()
    ref_frame.FrameOfReferenceUID = ct_first.FrameOfReferenceUID
    ref_study = Dataset()
    ref_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    ref_study.ReferencedSOPInstanceUID = ct_first.StudyInstanceUID
    ref_series = Dataset()
    ref_series.SeriesInstanceUID = ct_first.SeriesInstanceUID
    contour_image_seq = Sequence()
    for sop in ct_sops:
        img_ref = Dataset()
        img_ref.ReferencedSOPClassUID = CTImageStorage
        img_ref.ReferencedSOPInstanceUID = sop
        contour_image_seq.append(img_ref)
    ref_series.ContourImageSequence = contour_image_seq
    ref_study.RTReferencedSeriesSequence = Sequence([ref_series])
    ref_frame.RTReferencedStudySequence = Sequence([ref_study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([ref_frame])
    ds.StructureSetROISequence = Sequence()
    ds.ROIContourSequence = Sequence()
    ds.RTROIObservationsSequence = Sequence()
    entries = _structure_entries(case_dir)
    dicom_name_map = {}
    meta_path = case_dir / "StructureSet_MetaData.json"
    if meta_path.exists():
        try:
            meta = _read_json(meta_path)
            for item in meta:
                key = item.get("name")
                if key:
                    dicom_name_map[key] = item.get("dicom_structure_name") or key
        except json.JSONDecodeError:
            pass
    roi_number = 1
    for entry in entries:
        name = entry["name"]
        mask_path = entry["path"]
        dataset = entry["dataset"]
        dicom_name = dicom_name_map.get(name, name)
        with h5py.File(mask_path, "r") as handle:
            mask = handle[dataset][:]
        if np.max(mask) <= 0:
            continue
        roi = Dataset()
        roi.ROINumber = roi_number
        roi.ReferencedFrameOfReferenceUID = ct_first.FrameOfReferenceUID
        roi.ROIName = dicom_name
        roi.ROIGenerationAlgorithm = "AUTOMATIC"
        ds.StructureSetROISequence.append(roi)
        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = roi_number
        color = STRUCTURE_COLORS[(roi_number - 1) % len(STRUCTURE_COLORS)]
        roi_contour.ROIDisplayColor = [int(color[0]), int(color[1]), int(color[2])]
        contour_seq = Sequence()
        for z in range(mask.shape[0]):
            slice_mask = mask[z].astype(bool)
            if not np.any(slice_mask):
                continue
            contours = measure.find_contours(slice_mask.astype(np.uint8), 0.5)
            if not contours:
                continue
            ipp = np.array(ct_positions[z], dtype=float)
            for contour in contours:
                points = []
                for row, col in contour:
                    coord = (
                        ipp
                        + row * float(row_spacing) * row_dir
                        + col * float(col_spacing) * col_dir
                    )
                    points.extend([float(coord[0]), float(coord[1]), float(coord[2])])
                if len(points) < 6:
                    continue
                if points[0:3] != points[-3:]:
                    points.extend(points[0:3])
                contour_item = Dataset()
                contour_item.ContourGeometricType = "CLOSED_PLANAR"
                contour_item.NumberOfContourPoints = int(len(points) / 3)
                contour_item.ContourData = points
                img_ref = Dataset()
                img_ref.ReferencedSOPClassUID = CTImageStorage
                img_ref.ReferencedSOPInstanceUID = ct_sops[z]
                contour_item.ContourImageSequence = Sequence([img_ref])
                contour_seq.append(contour_item)
        if not contour_seq:
            continue
        roi_contour.ContourSequence = contour_seq
        ds.ROIContourSequence.append(roi_contour)
        obs = Dataset()
        obs.ObservationNumber = roi_number
        obs.ReferencedROINumber = roi_number
        obs.RTROIInterpretedType = "ORGAN"
        obs.ROIInterpreter = ""
        ds.RTROIObservationsSequence.append(obs)
        roi_number += 1
    ds.save_as(str(rt_path), write_like_original=False)
    return {
        "status": "created",
        "artifact": str(rt_path),
        "rtstruct_path": str(rt_path),
        "ct_dir": str(ct_result.get("ct_dir")),
    }

def _write_rt_dose_dicom(
    out_dir: Path,
    case_dir: Path,
    plan_ds,
    overwrite: bool = False,
) -> dict:
    dose_dcm_path = out_dir / "rt_dose_portpy_vmat.dcm"
    if dose_dcm_path.exists() and not overwrite:
        return {"status": "exists", "artifact": dose_dcm_path.name}
    dose_path = out_dir / "dose_3d.npy"
    if not dose_path.exists():
        raise HTTPException(status_code=404, detail="dose_3d.npy not found")
    meta_path = out_dir / "dose_3d_meta.json"
    dose_arr = np.load(dose_path)
    ct_meta, ct_path, dataset = _ct_reference(case_dir)
    with h5py.File(ct_path, "r") as handle:
        ct_shape = handle[dataset].shape
    if tuple(dose_arr.shape) != tuple(ct_shape):
        raise HTTPException(
            status_code=400,
            detail=f"dose_3d shape {dose_arr.shape} does not match CT shape {ct_shape}",
        )
    if meta_path.exists():
        meta = _read_json(meta_path)
    else:
        meta = ct_meta
    z_slices, rows, cols = dose_arr.shape
    max_dose = float(np.max(dose_arr))
    max_pixel = np.iinfo(np.uint32).max
    if max_dose <= 0:
        scaling = 1.0
        scaled = np.zeros_like(dose_arr, dtype=np.uint32)
    else:
        scaling = max_dose / max_pixel
        scaled = np.rint(dose_arr / scaling)
        scaled = np.clip(scaled, 0, max_pixel).astype(np.uint32)
    origin = meta.get("origin_xyz_mm", [0.0, 0.0, 0.0])
    spacing = meta.get("resolution_xyz_mm", [1.0, 1.0, 1.0])
    direction = meta.get("direction") or [1, 0, 0, 0, 1, 0, 0, 0, 1]
    orientation = [
        float(direction[0]),
        float(direction[1]),
        float(direction[2]),
        float(direction[3]),
        float(direction[4]),
        float(direction[5]),
    ]
    try:
        from pydicom import dcmread
        from pydicom.dataset import Dataset
        from pydicom.sequence import Sequence
        from pydicom.uid import generate_uid
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"pydicom not available: {exc}") from exc
    template_path = _rt_dose_template(case_dir)
    ds = dcmread(str(template_path))
    ds.Rows = int(rows)
    ds.Columns = int(cols)
    ds.NumberOfFrames = int(z_slices)
    ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
    ds.GridFrameOffsetVector = [float(i * spacing[2]) for i in range(z_slices)]
    ds.ImagePositionPatient = [float(origin[0]), float(origin[1]), float(origin[2])]
    ds.ImageOrientationPatient = orientation
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseGridScaling = float(scaling)
    ds.StudyInstanceUID = getattr(plan_ds, "StudyInstanceUID", ds.StudyInstanceUID)
    ds.FrameOfReferenceUID = getattr(plan_ds, "FrameOfReferenceUID", ds.FrameOfReferenceUID)
    ds.PatientName = getattr(plan_ds, "PatientName", ds.get("PatientName", ""))
    ds.PatientID = getattr(plan_ds, "PatientID", ds.get("PatientID", ""))
    ds.StudyDate = getattr(plan_ds, "StudyDate", ds.get("StudyDate", ""))
    ds.StudyTime = getattr(plan_ds, "StudyTime", ds.get("StudyTime", ""))
    dose_series_uid = generate_uid()
    dose_sop_uid = generate_uid()
    _update_dicom_uids(ds, dose_series_uid, dose_sop_uid)
    if hasattr(ds, "ReferencedRTPlanSequence") and ds.ReferencedRTPlanSequence:
        ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = plan_ds.SOPInstanceUID
    else:
        ref = Sequence()
        item = Dataset()
        item.ReferencedSOPClassUID = plan_ds.SOPClassUID
        item.ReferencedSOPInstanceUID = plan_ds.SOPInstanceUID
        ref.append(item)
        ds.ReferencedRTPlanSequence = ref
    ds.PixelData = np.ascontiguousarray(scaled).tobytes()
    ds.save_as(str(dose_dcm_path), write_like_original=False)
    return {"status": "created", "artifact": dose_dcm_path.name}

def _mask_edge(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    edge = mask & (
        (~padded[:-2, 1:-1])
        | (~padded[2:, 1:-1])
        | (~padded[1:-1, :-2])
        | (~padded[1:-1, 2:])
    )
    return edge


def _dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= padded[
                radius + dy : radius + dy + mask.shape[0],
                radius + dx : radius + dx + mask.shape[1],
            ]
    return out


def _window_ct(slice_hu: np.ndarray, window: float, level: float) -> np.ndarray:
    if window <= 0:
        window = 1.0
    min_val = level - window / 2.0
    max_val = level + window / 2.0
    clipped = np.clip(slice_hu, min_val, max_val)
    scaled = (clipped - min_val) / (max_val - min_val)
    return (scaled * 255.0).astype(np.uint8)


def _colorize_dose(slice_gy: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (slice_gy - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    xp = np.array([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float32)
    r = np.interp(norm, xp, [0, 0, 41, 255, 255])
    g = np.interp(norm, xp, [0, 128, 220, 200, 64])
    b = np.interp(norm, xp, [0, 255, 255, 64, 0])
    alpha = np.where(norm > 0, (norm ** 0.7) * 255.0, 0.0)
    rgba = np.stack([r, g, b, alpha], axis=-1)
    return rgba.astype(np.uint8)


def _run_worker(run_id: str, req: RunRequest, out_dir: Path) -> None:
    adapter = None
    if req.adapter == "example":
        adapter = ExampleAdapter(data_root=_default_data_dir())
    elif req.adapter == "huggingface":
        if not req.hf_repo_id:
            _write_json(
                out_dir / "status.json",
                {"state": "error", "error": "hf_repo_id is required for huggingface adapter"},
            )
            return
        adapter = HuggingFaceAdapter(
            repo_id=req.hf_repo_id,
            cache_dir=Path(__file__).resolve().parents[1] / "data" / "raw" / "huggingface",
            token=req.hf_token,
            subdir=req.hf_subdir,
        )

    use_planner_beams = req.use_planner_beams or req.fast or req.super_fast
    use_available_beams = req.use_available_beams or req.fast or req.super_fast
    force_sparse = req.force_sparse or req.fast or req.super_fast

    try:
        if req.optimizer == "compressrtp":
            run_compressrtp(
                case_id=req.case_id,
                protocol_name=req.protocol,
                data_dir=_default_data_dir(),
                out_dir=out_dir,
                solver="MOSEK",
                use_planner_beams=use_planner_beams,
                use_available_beams=use_available_beams,
                beam_ids_override=req.beam_ids,
                compress_mode=req.compress_mode,
                threshold_perc=req.threshold_perc,
                rank=req.rank,
                step=req.step,
                fast=req.fast,
                super_fast=req.super_fast,
                use_gpu=req.use_gpu,
                tag=req.tag,
                adapter=adapter,
            )
        else:
            run_echo_example(
                case_id=req.case_id,
                protocol_name=req.protocol,
                data_dir=_default_data_dir(),
                out_dir=out_dir,
                use_planner_beams=use_planner_beams,
                use_available_beams=use_available_beams,
                force_sparse=force_sparse,
                super_fast=req.super_fast,
                tag=req.tag,
                adapter=adapter,
            )
    except Exception as exc:
        _write_json(out_dir / "status.json", {"state": "error", "error": str(exc)})


def _batch_worker(
    batch_id: str, run_payloads: list[tuple[str, RunRequest, Path]], label: Optional[str]
) -> None:
    run_ids = [run_id for run_id, _req, _out_dir in run_payloads]
    total = len(run_payloads)
    for idx, (run_id, req, out_dir) in enumerate(run_payloads):
        _write_batch_status(
            batch_id,
            {
                "batch_id": batch_id,
                "label": label,
                "state": "running",
                "current_index": idx,
                "current_run_id": run_id,
                "run_ids": run_ids,
                "total": total,
            },
        )
        _run_worker(run_id, req, out_dir)
    _write_batch_status(
        batch_id,
        {
            "batch_id": batch_id,
            "label": label,
            "state": "completed",
            "current_index": total - 1 if total else None,
            "current_run_id": run_ids[-1] if run_ids else None,
            "run_ids": run_ids,
            "total": total,
        },
    )


@app.post("/runs", response_model=RunResponse)
def create_run(req: RunRequest) -> RunResponse:
    run_id = _safe_run_id(req.case_id, req.optimizer)
    out_dir = RUNS_DIR_COMPRESS / run_id if req.optimizer == "compressrtp" else RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_dir / "status.json",
        {"state": "queued", "case_id": req.case_id, "tag": req.tag, "optimizer": req.optimizer},
    )

    thread = threading.Thread(target=_run_worker, args=(run_id, req, out_dir), daemon=True)
    thread.start()
    return RunResponse(run_id=run_id)


@app.post("/runs/batch", response_model=BatchResponse)
def create_batch(req: BatchRequest) -> BatchResponse:
    if not req.runs:
        raise HTTPException(status_code=400, detail="Batch must include at least one run request")
    batch_id = uuid4().hex
    run_payloads: list[tuple[str, RunRequest, Path]] = []
    run_ids: list[str] = []
    for run_req in req.runs:
        effective_tag = run_req.tag or req.label
        if effective_tag != run_req.tag:
            run_req = run_req.model_copy(update={"tag": effective_tag})
        run_id = _safe_run_id(run_req.case_id, run_req.optimizer)
        out_dir = RUNS_DIR_COMPRESS / run_id if run_req.optimizer == "compressrtp" else RUNS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            out_dir / "status.json",
            {"state": "queued", "case_id": run_req.case_id, "tag": run_req.tag, "optimizer": run_req.optimizer},
        )
        run_ids.append(run_id)
        run_payloads.append((run_id, run_req, out_dir))
    _write_batch_status(
        batch_id,
        {
            "batch_id": batch_id,
            "label": req.label,
            "state": "queued",
            "current_index": None,
            "current_run_id": None,
            "run_ids": run_ids,
            "total": len(run_payloads),
        },
    )
    thread = threading.Thread(
        target=_batch_worker,
        args=(batch_id, run_payloads, req.label),
        daemon=True,
    )
    thread.start()
    return BatchResponse(batch_id=batch_id, run_ids=run_ids)


@app.get("/runs/batch/{batch_id}")
def get_batch(batch_id: str) -> dict:
    return _read_batch_status(batch_id)


@app.get("/runs")
def list_runs() -> dict:
    runs = []
    for run_type, root in _run_roots().items():
        for path in root.iterdir():
            if not path.is_dir():
                continue
            status_path = path / "status.json"
            status = _read_json(status_path) if status_path.exists() else {"state": "unknown"}
            config_path = path / "config.json"
            config = _read_json(config_path) if config_path.exists() else {}
            plan_score_path = path / "plan_score.json"
            plan_score = None
            if plan_score_path.exists():
                plan_payload = _read_json(plan_score_path)
                plan_score = {
                    "plan_score": plan_payload.get("plan_score"),
                    "plan_percentile": plan_payload.get("plan_percentile"),
                }
            runs.append(
                {
                    "run_id": path.name,
                    "status": status,
                    "run_type": config.get("optimizer", run_type),
                    "case_id": config.get("case_id"),
                    "protocol": config.get("protocol"),
                    "tag": config.get("tag") or status.get("tag"),
                    "started_at": status.get("started_at"),
                    "timestamp": path.stat().st_mtime,
                    "plan_score": plan_score,
                }
            )
    runs.sort(key=lambda item: item.get("timestamp", 0), reverse=True)
    return {"runs": runs}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    out_dir = _find_run_dir(run_id)
    status_path = out_dir / "status.json"
    status = _read_json(status_path) if status_path.exists() else {"state": "unknown"}
    artifacts = sorted([p.name for p in out_dir.iterdir() if p.is_file()])
    config_path = out_dir / "config.json"
    config = _read_json(config_path) if config_path.exists() else {}
    return {
        "run_id": run_id,
        "status": status,
        "artifacts": artifacts,
        "run_type": config.get("optimizer", "echo-vmat"),
        "case_id": config.get("case_id"),
        "protocol": config.get("protocol"),
        "tag": config.get("tag") or status.get("tag"),
    }


@app.get("/patients")
def list_patients(protocol: str = "Lung_2Gy_30Fx") -> dict:
    try:
        try:
            from backend.plan_score.score import compute_population_summary
        except ImportError:
            from plan_score.score import compute_population_summary
        summary = compute_population_summary(protocol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    patients = [item.get("case_id") for item in summary.get("patients", []) if item.get("case_id")]
    return {"protocol": protocol, "patients": patients}


@app.get("/plan-score/population")
def get_population_plan_scores(protocol: str = "Lung_2Gy_30Fx") -> dict:
    try:
        try:
            from backend.plan_score.score import compute_population_summary
        except ImportError:
            from plan_score.score import compute_population_summary
        return compute_population_summary(protocol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/runs/{run_id}/plan-score")
def get_plan_score(run_id: str) -> dict:
    out_dir = _find_run_dir(run_id)
    config = _load_run_config(run_id)
    protocol = config.get("protocol") or "Lung_2Gy_30Fx"
    plan_score_path = out_dir / "plan_score.json"
    if plan_score_path.exists():
        return _read_json(plan_score_path)
    try:
        try:
            from backend.plan_score.score import compute_plan_score_for_run
        except ImportError:
            from plan_score.score import compute_plan_score_for_run
        plan_score = compute_plan_score_for_run(out_dir, protocol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _write_json(plan_score_path, plan_score)
    return plan_score


@app.get("/runs/{run_id}/events")
async def stream_events(run_id: str) -> StreamingResponse:
    out_dir = _find_run_dir(run_id)
    events_path = out_dir / "events.jsonl"
    status_path = out_dir / "status.json"

    async def event_stream():
        last_pos = 0
        while True:
            if events_path.exists():
                with events_path.open("r", encoding="utf-8") as handle:
                    handle.seek(last_pos)
                    while True:
                        line = handle.readline()
                        if not line:
                            break
                        last_pos = handle.tell()
                        yield f"data: {line.strip()}\n\n"
            if status_path.exists():
                status = _read_json(status_path)
                if status.get("state") in {"completed", "error"}:
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/runs/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str) -> StreamingResponse:
    out_dir = _find_run_dir(run_id)
    if Path(name).name != name:
        raise HTTPException(status_code=400, detail="invalid artifact name")
    path = out_dir / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    media_type, _ = mimetypes.guess_type(path.name)
    return StreamingResponse(
        _iter_file(path),
        media_type=media_type or "application/octet-stream",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/runs/{run_id}/ct/meta")
def get_ct_meta(run_id: str) -> dict:
    case_dir, _config = _resolve_case_dir(run_id)
    meta, ct_path, dataset = _ct_reference(case_dir)
    with h5py.File(ct_path, "r") as handle:
        shape = list(handle[dataset].shape)
    return {
        "shape_zyx": shape,
        "slice_count": shape[0] if shape else 0,
        "origin_xyz_mm": meta.get("origin_xyz_mm"),
        "resolution_xyz_mm": meta.get("resolution_xyz_mm"),
        "direction": meta.get("direction"),
    }


@app.get("/runs/{run_id}/ct/slice")
def get_ct_slice(
    run_id: str,
    index: int = Query(0, ge=0),
    window: float = Query(400.0, gt=0),
    level: float = Query(40.0),
) -> StreamingResponse:
    case_dir, _config = _resolve_case_dir(run_id)
    _meta, ct_path, dataset = _ct_reference(case_dir)
    with h5py.File(ct_path, "r") as handle:
        ct = handle[dataset]
        max_index = ct.shape[0] - 1
        slice_index = min(index, max_index)
        slice_hu = ct[slice_index, :, :]
    slice_u8 = _window_ct(slice_hu, window=window, level=level)
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pillow not installed: {exc}") from exc
    image = Image.fromarray(slice_u8, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = buffer.getvalue()
    return StreamingResponse(
        iter([content]),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/runs/{run_id}/structures")
def list_structures(run_id: str) -> dict:
    case_dir, _config = _resolve_case_dir(run_id)
    entries = _structure_entries(case_dir)
    return {"structures": [entry["name"] for entry in entries]}


@app.get("/runs/{run_id}/structures/slice")
def get_structure_slice(
    run_id: str,
    index: int = Query(0, ge=0),
    names: Optional[str] = Query(None),
) -> StreamingResponse:
    case_dir, _config = _resolve_case_dir(run_id)
    entries = _structure_entries(case_dir)
    name_filter = None
    if names:
        name_filter = {name.strip().upper() for name in names.split(",") if name.strip()}
    selected = [
        entry for entry in entries if name_filter is None or entry["name"].upper() in name_filter
    ]
    if not selected:
        raise HTTPException(status_code=404, detail="No matching structures found")
    overlay = None
    color_idx = 0
    entries_by_path = {}
    for entry in selected:
        entries_by_path.setdefault(entry["path"], []).append(entry)
    for path, path_entries in entries_by_path.items():
        with h5py.File(path, "r") as handle:
            for entry in path_entries:
                data = handle[entry["dataset"]]
                max_index = data.shape[0] - 1
                slice_index = min(index, max_index)
                mask_slice = data[slice_index, :, :]
                mask = mask_slice > 0
                edge = _dilate_mask(_mask_edge(mask), radius=2)
                if overlay is None:
                    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                color = STRUCTURE_COLORS[color_idx % len(STRUCTURE_COLORS)]
                overlay[edge, 0] = color[0]
                overlay[edge, 1] = color[1]
                overlay[edge, 2] = color[2]
                overlay[edge, 3] = 230
                color_idx += 1
    if overlay is None:
        raise HTTPException(status_code=404, detail="No structure overlay generated")
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pillow not installed: {exc}") from exc
    image = Image.fromarray(overlay, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = buffer.getvalue()
    return StreamingResponse(
        iter([content]),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/runs/{run_id}/rtplan")
def create_rt_plan(run_id: str, overwrite: bool = Query(False)) -> dict:
    out_dir = _find_run_dir(run_id)
    rt_path = out_dir / "rt_plan_portpy_vmat.dcm"
    try:
        import portpy.photon as pp
        from portpy.photon.utils import write_rt_plan_vmat
        from pydicom import dcmread
        from pydicom.uid import generate_uid
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"portpy/pydicom not available: {exc}") from exc
    plan_path = out_dir / "my_plan.pkl"
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="my_plan.pkl not found")
    case_dir, _config = _resolve_case_dir(run_id)
    template_path = _rt_plan_template(case_dir)
    _ensure_echo_vmat_on_path()
    plan_created = False
    if rt_path.exists() and not overwrite:
        plan_ds = dcmread(str(rt_path))
    else:
        my_plan = pp.load_plan(plan_name=plan_path.name, path=str(out_dir))
        write_rt_plan_vmat(
            my_plan=my_plan,
            in_rt_plan_file=str(template_path),
            out_rt_plan_file=str(rt_path),
        )
        plan_ds = dcmread(str(rt_path))
        _update_dicom_uids(plan_ds, generate_uid(), generate_uid())
        plan_ds.save_as(str(rt_path), write_like_original=False)
        plan_created = True
    dose_path = out_dir / "dose_3d.npy"
    if not dose_path.exists() or overwrite:
        create_dose_3d(run_id=run_id, overwrite=overwrite)
    dose_payload = _write_rt_dose_dicom(out_dir, case_dir, plan_ds, overwrite=overwrite)
    status = "created" if plan_created or dose_payload.get("status") == "created" else "exists"
    return {
        "status": status,
        "artifact": rt_path.name,
        "dose_artifact": dose_payload.get("artifact"),
        "rt_plan_path": str(rt_path),
        "rt_dose_path": str(out_dir / dose_payload.get("artifact"))
        if dose_payload.get("artifact")
        else None,
    }


@app.post("/runs/{run_id}/ct-dicom")
def create_ct_dicom(run_id: str, overwrite: bool = Query(False)) -> dict:
    case_dir, config = _resolve_case_dir(run_id)
    case_id = config.get("case_id") or case_dir.name
    return _write_ct_dicom(case_dir, case_id, overwrite=overwrite)


@app.post("/runs/{run_id}/rtstruct")
def create_rt_struct(run_id: str, overwrite: bool = Query(False)) -> dict:
    case_dir, config = _resolve_case_dir(run_id)
    case_id = config.get("case_id") or case_dir.name
    return _write_rt_struct_dicom(case_dir, case_id, overwrite=overwrite)


@app.post("/runs/{run_id}/dose-3d")
def create_dose_3d(run_id: str, overwrite: bool = Query(False)) -> dict:
    out_dir = _find_run_dir(run_id)
    dose_path = out_dir / "dose_3d.npy"
    meta_path = out_dir / "dose_3d_meta.json"
    if dose_path.exists() and not overwrite:
        return {"status": "exists", "artifact": dose_path.name}
    try:
        import portpy.photon as pp
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"portpy not available: {exc}") from exc
    plan_path = out_dir / "my_plan.pkl"
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="my_plan.pkl not found")
    sol_paths = list(out_dir.glob("sol_step*.pkl"))
    if not sol_paths:
        raise HTTPException(status_code=404, detail="No sol_step*.pkl found for run")
    sol_paths.sort(
        key=lambda path: int(re.search(r"sol_step(\\d+)", path.stem).group(1))
        if re.search(r"sol_step(\\d+)", path.stem)
        else -1
    )
    sol_path = sol_paths[-1]
    _ensure_echo_vmat_on_path()
    my_plan = pp.load_plan(plan_name=plan_path.name, path=str(out_dir))
    sol = pp.load_optimal_sol(sol_name=sol_path.name, path=str(out_dir))
    if "act_dose_v" in sol:
        dose_1d = sol["act_dose_v"] * my_plan.get_num_of_fractions()
    else:
        intensity = sol.get("optimal_intensity")
        if intensity is None:
            raise HTTPException(status_code=500, detail="Solution missing act_dose_v and optimal_intensity")
        dose_1d = my_plan.inf_matrix.A @ intensity * my_plan.get_num_of_fractions()
    dose_3d = my_plan.inf_matrix.dose_1d_to_3d(dose_1d=dose_1d).astype(np.float32)
    np.save(dose_path, dose_3d, allow_pickle=False)
    dose_min = float(np.min(dose_3d))
    dose_max = float(np.max(dose_3d))
    case_dir, config = _resolve_case_dir(run_id)
    meta, _ct_path, _dataset = _ct_reference(case_dir)
    _write_json(
        meta_path,
        {
            "run_id": run_id,
            "case_id": config.get("case_id"),
            "shape_zyx": list(dose_3d.shape),
            "origin_xyz_mm": meta.get("origin_xyz_mm"),
            "resolution_xyz_mm": meta.get("resolution_xyz_mm"),
            "direction": meta.get("direction"),
            "units": "Gy",
            "source_solution": sol_path.name,
            "min_gy": dose_min,
            "max_gy": dose_max,
        },
    )
    return {"status": "created", "artifact": dose_path.name}


@app.get("/runs/{run_id}/dose/slice")
def get_dose_slice(
    run_id: str,
    index: int = Query(0, ge=0),
    dose_min: float = Query(0.0, ge=0.0),
    dose_max: Optional[float] = Query(None),
) -> StreamingResponse:
    out_dir = _find_run_dir(run_id)
    dose_path = out_dir / "dose_3d.npy"
    if not dose_path.exists():
        raise HTTPException(status_code=404, detail="dose_3d.npy not found")
    meta_path = out_dir / "dose_3d_meta.json"
    if dose_max is None and meta_path.exists():
        meta = _read_json(meta_path)
        dose_max = meta.get("max_gy")
    dose_arr = np.load(dose_path, mmap_mode="r")
    max_index = dose_arr.shape[0] - 1
    slice_index = min(index, max_index)
    slice_gy = dose_arr[slice_index, :, :]
    if dose_max is None:
        dose_max = float(np.max(slice_gy))
    rgba = _colorize_dose(slice_gy, vmin=dose_min, vmax=dose_max)
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pillow not installed: {exc}") from exc
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = buffer.getvalue()
    return StreamingResponse(
        iter([content]),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
