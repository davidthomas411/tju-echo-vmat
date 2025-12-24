from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp


@dataclass
class GpuContext:
    enabled: bool
    reason: str | None = None
    free_mb: float | None = None
    total_mb: float | None = None
    name: str | None = None


def get_gpu_context(use_gpu: bool) -> tuple[GpuContext, dict[str, Any]]:
    if not use_gpu:
        return GpuContext(enabled=False, reason="disabled"), {}
    try:
        import cupy as cp
        import cupyx
    except Exception as exc:  # pragma: no cover - depends on optional install
        return GpuContext(enabled=False, reason=str(exc)), {}
    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        props = cp.cuda.runtime.getDeviceProperties(0)
    except Exception as exc:  # pragma: no cover - depends on driver/runtime
        return GpuContext(enabled=False, reason=str(exc)), {}
    name = props.get("name", b"")
    if isinstance(name, (bytes, bytearray)):
        name = name.decode("utf-8", errors="replace")
    ctx = GpuContext(
        enabled=True,
        free_mb=free_bytes / (1024 * 1024),
        total_mb=total_bytes / (1024 * 1024),
        name=name or None,
    )
    return ctx, {"xp": cp, "sparse": cupyx.scipy.sparse}


def estimate_chunk_rows(A: np.ndarray | sp.spmatrix, max_chunk_mb: int | None) -> int | None:
    if max_chunk_mb is None:
        return None
    if not hasattr(A, "shape") or not hasattr(A, "dtype"):
        return None
    n_cols = int(A.shape[1])
    bytes_per_row = n_cols * np.dtype(A.dtype).itemsize
    if bytes_per_row <= 0:
        return None
    chunk_rows = int((max_chunk_mb * 1024 * 1024) / bytes_per_row)
    return max(1, chunk_rows)


def gpu_matmul(
    A: np.ndarray | sp.spmatrix,
    x: np.ndarray,
    xp: Any,
    sparse_module: Any,
    max_chunk_mb: int | None = None,
    chunk_rows: int | None = None,
) -> np.ndarray:
    if chunk_rows is None:
        chunk_rows = estimate_chunk_rows(A, max_chunk_mb)

    if chunk_rows is None or chunk_rows >= A.shape[0]:
        if sp.issparse(A):
            A_gpu = sparse_module.csr_matrix(A)
            x_gpu = xp.asarray(x)
            y_gpu = A_gpu.dot(x_gpu)
        else:
            A_gpu = xp.asarray(A)
            x_gpu = xp.asarray(x)
            y_gpu = A_gpu @ x_gpu
        return xp.asnumpy(y_gpu)

    x_gpu = xp.asarray(x)
    n_rows = int(A.shape[0])
    out = np.empty(n_rows, dtype=np.result_type(A.dtype, x.dtype))
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        block = A[start:end]
        if sp.issparse(block):
            block_gpu = sparse_module.csr_matrix(block)
            y_gpu = block_gpu.dot(x_gpu)
        else:
            block_gpu = xp.asarray(block)
            y_gpu = block_gpu @ x_gpu
        out[start:end] = xp.asnumpy(y_gpu)
    return out
