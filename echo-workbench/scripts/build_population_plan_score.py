from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKBENCH_DIR = Path(__file__).resolve().parents[1]
if str(WORKBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_DIR))

from backend.plan_score.population import build_population_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build population metrics for plan score.")
    parser.add_argument("--protocol", default="Lung_2Gy_30Fx")
    parser.add_argument(
        "--data-root",
        default=str(WORKBENCH_DIR / "data" / "raw" / "huggingface" / "PortPy_Dataset" / "data"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(WORKBENCH_DIR / "data" / "processed" / "plan-score" / "Lung_2Gy_30Fx"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    population = build_population_metrics(
        protocol_name=args.protocol,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        limit=args.limit,
    )
    print(f"Saved population metrics for {len(population['patients'])} patients.")


if __name__ == "__main__":
    main()
