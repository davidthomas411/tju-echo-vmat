from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Objective:
    id: str
    label: str
    structure: str
    metric_type: str
    parameters: dict[str, Any]
    unit: str
    direction: str
    goal: float | None
    limit: float | None
    priority: int


def _criteria_path(protocol_name: str) -> Path:
    workbench_dir = Path(__file__).resolve().parents[2]
    config_root = workbench_dir / "echo-vmat" / "echo_vmat" / "config_files"
    return config_root / f"{protocol_name}_clinical_criteria.json"


def _priority_from_weight(weight: float | None) -> int:
    if weight is None:
        return 3
    if weight >= 100:
        return 1
    if weight >= 50:
        return 2
    return 3


def _build_objective_id(metric_type: str, structure: str, dose_gy: float | None = None) -> str:
    if dose_gy is None:
        return f"{metric_type}.{structure}"
    return f"{metric_type}.{structure}.{dose_gy:g}Gy"


def _label_for(metric_type: str, structure: str, dose_gy: float | None = None) -> str:
    if metric_type == "max_dose":
        return f"Max Dose ({structure})"
    if metric_type == "mean_dose":
        return f"Mean Dose ({structure})"
    if metric_type == "dose_volume_V":
        return f"V{dose_gy:g}Gy ({structure})"
    if metric_type == "coverage_D":
        return f"D95 (PTV)"
    return f"{metric_type} ({structure})"


def load_objectives(protocol_name: str) -> list[Objective]:
    criteria_path = _criteria_path(protocol_name)
    if not criteria_path.exists():
        raise FileNotFoundError(f"Clinical criteria file not found: {criteria_path}")

    with criteria_path.open("r", encoding="utf-8") as handle:
        criteria = json.load(handle)

    objectives: dict[tuple[str, str, float | None], Objective] = {}
    for item in criteria.get("criteria", []):
        metric_type = item.get("type")
        params = item.get("parameters", {})
        constraints = item.get("constraints", {})
        structure = str(params.get("structure_name", "")).upper()
        if not structure:
            continue
        weight = params.get("weight")
        priority = _priority_from_weight(weight)

        goal = constraints.get("goal_dose_gy")
        limit = constraints.get("limit_dose_gy")
        unit = "Gy"
        dose_gy = None
        direction = "lower"

        if metric_type == "dose_volume_V":
            dose_gy = float(params.get("dose_gy"))
            goal = constraints.get("goal_volume_perc")
            limit = constraints.get("limit_volume_perc")
            unit = "%"
            direction = "lower"
        elif metric_type == "max_dose":
            direction = "lower"
        elif metric_type == "mean_dose":
            direction = "lower"
        else:
            continue

        key = (metric_type, structure, dose_gy)
        obj = Objective(
            id=_build_objective_id(metric_type, structure, dose_gy),
            label=_label_for(metric_type, structure, dose_gy),
            structure=structure,
            metric_type=metric_type,
            parameters={"dose_gy": dose_gy} if dose_gy is not None else {},
            unit=unit,
            direction=direction,
            goal=goal,
            limit=limit,
            priority=priority,
        )
        existing = objectives.get(key)
        if existing:
            if existing.goal is None and obj.goal is not None:
                objectives[key] = obj
            continue
        objectives[key] = obj

    pres_per_fraction = float(criteria.get("pres_per_fraction_gy", 0))
    num_fractions = float(criteria.get("num_of_fractions", 0))
    prescription = pres_per_fraction * num_fractions
    coverage = Objective(
        id="coverage_D95.PTV",
        label=_label_for("coverage_D", "PTV"),
        structure="PTV",
        metric_type="coverage_D",
        parameters={"coverage_perc": 95.0},
        unit="Gy",
        direction="higher",
        goal=prescription or None,
        limit=None,
        priority=1,
    )
    objectives[("coverage_D", "PTV", None)] = coverage

    return list(objectives.values())


def objectives_for_protocol(protocol_name: str) -> list[Objective]:
    return load_objectives(protocol_name)
