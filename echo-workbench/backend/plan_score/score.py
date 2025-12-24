from __future__ import annotations

import json
from bisect import bisect_left
from math import exp, log
from pathlib import Path
from typing import Any

from .metrics import compute_metric_from_dvh
from .objectives import Objective, objectives_for_protocol


def _processed_dir(protocol_name: str) -> Path:
    workbench_dir = Path(__file__).resolve().parents[2]
    return workbench_dir / "data" / "processed" / "plan-score" / protocol_name


def _load_population(protocol_name: str) -> dict[str, Any]:
    path = _processed_dir(protocol_name) / "population_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Population metrics not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sorted_values(population: dict[str, Any], objective: Objective) -> list[float]:
    values = []
    for patient in population.get("patients", []):
        metric = patient.get("metrics", {}).get(objective.id)
        if metric is None:
            continue
        values.append(float(metric))
    return sorted(values)


def _percentile_rank(sorted_values: list[float], value: float) -> float:
    if not sorted_values:
        return 0.0
    if value <= sorted_values[0]:
        return 0.0
    if value >= sorted_values[-1]:
        return 100.0
    idx = bisect_left(sorted_values, value)
    return float(idx) / float(len(sorted_values) - 1) * 100.0


def _percentile_value(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    clamped = max(0.0, min(100.0, percentile))
    idx = int(round((clamped / 100.0) * (len(sorted_values) - 1)))
    return float(sorted_values[idx])


def _score_from_percentile(percentile: float, direction: str) -> float:
    score = percentile
    if direction == "lower":
        score = 100.0 - percentile
    return max(0.0, min(100.0, score))


def _target_value(obj: Objective) -> float | None:
    return obj.goal if obj.goal is not None else obj.limit


def compute_plan_score(
    dvh: dict[str, dict],
    protocol_name: str,
) -> dict[str, Any]:
    objectives = objectives_for_protocol(protocol_name)
    population = _load_population(protocol_name)

    objective_rows: list[dict[str, Any]] = []
    scores = []
    for obj in objectives:
        struct_key = obj.structure
        dvh_entry = dvh.get(struct_key)
        value = compute_metric_from_dvh(obj, dvh_entry)
        sorted_vals = _sorted_values(population, obj)
        percentile = _percentile_rank(sorted_vals, value) if value is not None else None
        score = _score_from_percentile(percentile, obj.direction) if percentile is not None else None
        if score is not None:
            scores.append(max(score, 0.1) / 100.0)

        target_val = _target_value(obj)
        target_percentile = None
        if target_val is not None and sorted_vals:
            target_pct = _percentile_rank(sorted_vals, float(target_val))
            target_percentile = _score_from_percentile(target_pct, obj.direction)

        objective_rows.append(
            {
                "id": obj.id,
                "label": obj.label,
                "structure": obj.structure,
                "metric_type": obj.metric_type,
                "value": value,
                "unit": obj.unit,
                "direction": obj.direction,
                "percentile": score,
                "target_value": target_val,
                "target_percentile": target_percentile,
                "goal": obj.goal,
                "limit": obj.limit,
                "priority": obj.priority,
            }
        )

    plan_score = 0.0
    if scores:
        plan_score = exp(sum(log(value) for value in scores) / len(scores)) * 100.0

    population_scores = population.get("population_scores", [])
    plan_percentile = None
    if population_scores:
        sorted_scores = sorted(population_scores)
        plan_percentile = _percentile_rank(sorted_scores, plan_score)

    return {
        "protocol": protocol_name,
        "plan_score": plan_score,
        "plan_percentile": plan_percentile,
        "population_count": len(population.get("patients", [])),
        "objectives": objective_rows,
    }


def compute_plan_score_for_run(run_dir: Path, protocol_name: str) -> dict[str, Any]:
    dvh_path = run_dir / "dvh.json"
    if not dvh_path.exists():
        raise FileNotFoundError("dvh.json not found")
    with dvh_path.open("r", encoding="utf-8") as handle:
        dvh = json.load(handle)
    return compute_plan_score(dvh, protocol_name)


def compute_population_scores(
    population: dict[str, Any], objectives: list[Objective]
) -> list[float]:
    distributions = {obj.id: _sorted_values(population, obj) for obj in objectives}
    scores = []
    for patient in population.get("patients", []):
        values = patient.get("metrics", {})
        per_obj_scores = []
        for obj in objectives:
            value = values.get(obj.id)
            if value is None:
                continue
            percentile = _percentile_rank(distributions[obj.id], float(value))
            score = _score_from_percentile(percentile, obj.direction)
            per_obj_scores.append(max(score, 0.1) / 100.0)
        if not per_obj_scores:
            scores.append(0.0)
            continue
        scores.append(exp(sum(log(val) for val in per_obj_scores) / len(per_obj_scores)) * 100.0)
    return scores


def compute_population_summary(protocol_name: str) -> dict[str, Any]:
    objectives = objectives_for_protocol(protocol_name)
    population = _load_population(protocol_name)
    scores = compute_population_scores(population, objectives)
    patients = population.get("patients", [])
    sorted_scores = sorted(scores)

    records = []
    for patient, score in zip(patients, scores):
        records.append(
            {
                "case_id": patient.get("case_id"),
                "plan_score": score,
                "percentile": _percentile_rank(sorted_scores, score) if sorted_scores else None,
            }
        )

    mean_score = sum(scores) / len(scores) if scores else None
    stats = {
        "min": _percentile_value(sorted_scores, 0.0),
        "p10": _percentile_value(sorted_scores, 10.0),
        "p25": _percentile_value(sorted_scores, 25.0),
        "median": _percentile_value(sorted_scores, 50.0),
        "p75": _percentile_value(sorted_scores, 75.0),
        "p90": _percentile_value(sorted_scores, 90.0),
        "max": _percentile_value(sorted_scores, 100.0),
        "mean": mean_score,
    }

    return {
        "protocol": protocol_name,
        "population_count": len(patients),
        "population_scores": scores,
        "stats": stats,
        "patients": records,
    }


def compute_reference_plan_score(case_id: str, protocol_name: str) -> dict[str, Any]:
    objectives = objectives_for_protocol(protocol_name)
    population = _load_population(protocol_name)
    patients = population.get("patients", [])
    patient = next((item for item in patients if item.get("case_id") == case_id), None)
    if patient is None:
        raise FileNotFoundError(f"Case {case_id} not found in population metrics")

    distributions = {obj.id: _sorted_values(population, obj) for obj in objectives}
    objective_rows: list[dict[str, Any]] = []
    scores = []
    metrics = patient.get("metrics", {})

    for obj in objectives:
        value = metrics.get(obj.id)
        percentile = (
            _percentile_rank(distributions[obj.id], float(value))
            if value is not None and distributions[obj.id]
            else None
        )
        score = _score_from_percentile(percentile, obj.direction) if percentile is not None else None
        if score is not None:
            scores.append(max(score, 0.1) / 100.0)

        target_val = _target_value(obj)
        target_percentile = None
        if target_val is not None and distributions[obj.id]:
            target_pct = _percentile_rank(distributions[obj.id], float(target_val))
            target_percentile = _score_from_percentile(target_pct, obj.direction)

        objective_rows.append(
            {
                "id": obj.id,
                "label": obj.label,
                "structure": obj.structure,
                "metric_type": obj.metric_type,
                "value": value,
                "unit": obj.unit,
                "direction": obj.direction,
                "percentile": score,
                "target_value": target_val,
                "target_percentile": target_percentile,
                "goal": obj.goal,
                "limit": obj.limit,
                "priority": obj.priority,
            }
        )

    plan_score = 0.0
    if scores:
        plan_score = exp(sum(log(value) for value in scores) / len(scores)) * 100.0

    population_scores = population.get("population_scores") or compute_population_scores(population, objectives)
    sorted_scores = sorted(population_scores) if population_scores else []
    plan_percentile = _percentile_rank(sorted_scores, plan_score) if sorted_scores else None

    return {
        "protocol": protocol_name,
        "case_id": case_id,
        "plan_score": plan_score,
        "plan_percentile": plan_percentile,
        "population_count": len(patients),
        "objectives": objective_rows,
    }
