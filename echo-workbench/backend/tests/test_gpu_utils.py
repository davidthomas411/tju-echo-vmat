import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

WORKBENCH_DIR = Path(__file__).resolve().parents[2]
if str(WORKBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_DIR))

from backend.gpu import get_gpu_context, gpu_matmul


class TestGpuUtils(unittest.TestCase):
    def test_gpu_context_disabled(self) -> None:
        ctx, modules = get_gpu_context(False)
        self.assertFalse(ctx.enabled)
        self.assertEqual(modules, {})

    def test_gpu_matmul_matches_cpu(self) -> None:
        ctx, modules = get_gpu_context(True)
        if not ctx.enabled:
            self.skipTest("CuPy not available or GPU unavailable.")
        A = np.arange(12, dtype=np.float32).reshape(3, 4)
        x = np.arange(4, dtype=np.float32)
        expected = A @ x
        result = gpu_matmul(A, x, xp=modules["xp"], sparse_module=modules["sparse"])
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        A_sparse = sp.csr_matrix(A)
        result_sparse = gpu_matmul(A_sparse, x, xp=modules["xp"], sparse_module=modules["sparse"])
        np.testing.assert_allclose(result_sparse, expected, rtol=1e-6, atol=1e-6)
