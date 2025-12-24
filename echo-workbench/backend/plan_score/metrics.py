from __future__ import annotations

import numpy as np

from .objectives import Objective


def compute_metric_from_dose(
    objective: Objective, dose_gy: np.ndarray, masks: dict[str, np.ndarray]
) -> float | None:
    mask = masks.get(objective.structure)
    if mask is None:
        return None
    voxels = dose_gy[mask]
    if voxels.size == 0:
        return None

    if objective.metric_type == "max_dose":
        return float(np.max(voxels))
    if objective.metric_type == "mean_dose":
        return float(np.mean(voxels))
    if objective.metric_type == "dose_volume_V":
        dose_thr = float(objective.parameters.get("dose_gy", 0.0))
        return float(np.mean(voxels >= dose_thr) * 100.0)
    if objective.metric_type == "coverage_D":
        coverage = float(objective.parameters.get("coverage_perc", 95.0))
        return float(np.percentile(voxels, 100.0 - coverage))
    return None


def compute_metric_from_dvh(objective: Objective, dvh_entry: dict) -> float | None:
    if not dvh_entry:
        return None
    dose = np.asarray(dvh_entry.get("dose_gy") or [], dtype=float)
    volume = np.asarray(dvh_entry.get("volume_fraction") or [], dtype=float)
    if dose.size == 0 or volume.size == 0:
        return None

    if objective.metric_type == "max_dose":
        return float(np.max(dose))
    if objective.metric_type == "mean_dose":
        return float(np.trapz(volume, dose))
    if objective.metric_type == "dose_volume_V":
        dose_thr = float(objective.parameters.get("dose_gy", 0.0))
        frac = float(np.interp(dose_thr, dose, volume))
        return frac * 100.0
    if objective.metric_type == "coverage_D":
        target = float(objective.parameters.get("coverage_perc", 95.0)) / 100.0
        rev_volume = volume[::-1]
        rev_dose = dose[::-1]
        return float(np.interp(target, rev_volume, rev_dose))
    return None


def required_structures(objectives: list[Objective]) -> set[str]:
    return {obj.structure for obj in objectives if obj.structure}
