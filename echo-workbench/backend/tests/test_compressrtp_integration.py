import os
import sys
import time
import unittest
from pathlib import Path

WORKBENCH_DIR = Path(__file__).resolve().parents[2]
if str(WORKBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_DIR))

REPO_ROOT = WORKBENCH_DIR.parent
LICENSE_PATH = REPO_ROOT / "mosek.lic"
if LICENSE_PATH.exists():
    os.environ.setdefault("MOSEKLM_LICENSE_FILE", str(LICENSE_PATH))
    os.environ.setdefault("MOSEK_LICENSE_FILE", str(LICENSE_PATH))

from backend.runner import run_compressrtp, _default_data_dir


class CompressRTPIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = Path(_default_data_dir())
        cls.case_id = "Lung_Patient_11"
        case_dir = cls.data_dir / cls.case_id
        compress_root = WORKBENCH_DIR / "compressrtp"
        if not compress_root.exists():
            raise unittest.SkipTest("CompressRTP submodule not found")
        if not case_dir.exists():
            raise unittest.SkipTest(f"Case data not found: {case_dir}")
        cls.out_root = WORKBENCH_DIR / "backend" / "runs-compressrtp-tests"
        cls.out_root.mkdir(parents=True, exist_ok=True)

    def _run_mode(self, mode: str, threshold: float = 20.0, rank: int = 1) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_dir = self.out_root / f"{mode}-{ts}"
        run_compressrtp(
            case_id=self.case_id,
            protocol_name="Lung_2Gy_30Fx",
            data_dir=self.data_dir,
            out_dir=out_dir,
            beam_ids_override=[0, 1, 2],
            compress_mode=mode,
            threshold_perc=threshold,
            rank=rank,
            fast=True,
            super_fast=True,
        )
        return out_dir

    def _assert_outputs(self, out_dir: Path) -> None:
        required = [
            "config.json",
            "status.json",
            "timing.json",
            "metrics.json",
            "dvh.json",
            "dvh_steps.png",
            "my_plan.pkl",
            "sol_step1.pkl",
        ]
        for name in required:
            self.assertTrue((out_dir / name).exists(), f"missing {name}")

    def test_sparse_only(self) -> None:
        out_dir = self._run_mode("sparse-only", threshold=20.0, rank=1)
        self._assert_outputs(out_dir)

    def test_sparse_plus_low_rank(self) -> None:
        out_dir = self._run_mode("sparse-plus-low-rank", threshold=20.0, rank=1)
        self._assert_outputs(out_dir)

    def test_wavelet(self) -> None:
        out_dir = self._run_mode("wavelet", threshold=20.0, rank=1)
        self._assert_outputs(out_dir)


if __name__ == "__main__":
    unittest.main()
