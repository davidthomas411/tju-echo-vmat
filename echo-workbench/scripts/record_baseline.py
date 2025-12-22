#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_run_dir(run_id: str, roots: list[Path]) -> Path:
    for root in roots:
        candidate = root / run_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Run '{run_id}' not found under {', '.join(str(r) for r in roots)}")


def _hash_floats(values: list[float]) -> str:
    payload = ",".join(f"{value:.6g}" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dvh_summary(dvh: dict) -> dict:
    summary: dict[str, dict] = {}
    for structure, values in dvh.items():
        dose = [float(v) for v in values.get("dose_gy", [])]
        volume = [float(v) for v in values.get("volume_fraction", [])]
        if not dose or not volume:
            summary[structure] = {"dose_points": len(dose), "volume_points": len(volume)}
            continue
        summary[structure] = {
            "dose_points": len(dose),
            "volume_points": len(volume),
            "dose_min": min(dose),
            "dose_max": max(dose),
            "dose_mean": sum(dose) / len(dose),
            "volume_min": min(volume),
            "volume_max": max(volume),
            "volume_mean": sum(volume) / len(volume),
            "dose_hash": _hash_floats(dose),
            "volume_hash": _hash_floats(volume),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a baseline summary for a run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "docs" / "baselines"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_roots = [
        repo_root / "echo-workbench" / "backend" / "runs",
        repo_root / "echo-workbench" / "backend" / "runs-compressrtp",
    ]
    run_dir = _find_run_dir(args.run_id, run_roots)

    baseline: dict[str, object] = {
        "run_id": args.run_id,
        "run_dir": str(run_dir),
    }

    for name in ("config.json", "status.json", "timing.json"):
        path = run_dir / name
        if path.exists():
            baseline[name.replace(".json", "")] = _read_json(path)

    dvh_path = run_dir / "dvh.json"
    if dvh_path.exists():
        dvh = _read_json(dvh_path)
        baseline["dvh_summary"] = _dvh_summary(dvh)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = _read_json(metrics_path)
        baseline["metrics"] = metrics.get("metrics") or metrics.get("records") or []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.run_id}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(baseline, handle, indent=2)

    print(f"Baseline saved: {output_path}")


if __name__ == "__main__":
    main()
