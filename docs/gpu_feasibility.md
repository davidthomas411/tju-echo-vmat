# GPU Feasibility (Initial Notes)

This is a living checklist for evaluating GPU acceleration paths without changing the ECHO-VMAT solver logic.

## Observations
- CVXPY + MOSEK is CPU-based; MOSEK does not expose GPU acceleration for these conic problems.
- The heaviest steps are influence matrix construction and optimization.

## CompressRTP GPU Plan (Clear Path)
Goal: keep the **same algorithm and solver logic**, but accelerate the expensive linear algebra steps that are pure data transforms.

### Phase 0: Baseline + guardrails
- Record a CPU baseline for one patient (fast + full) with `timing.json` and `max_rss_mb`.
- Add a single flag (e.g., `--gpu`) that is **off by default** and only toggles GPU kernels.
- Define numeric tolerances for parity:
  - Dose difference < 1e-3 Gy (max abs)
  - DVH curves within 0.5% volume at matched dose

### Phase 1: GPU acceleration for pure matrix ops (no solver changes)
1) **Influence matrix operations** (CPU still builds A, GPU used for multiplies)
   - Convert `A` (sparse) to GPU CSR via CuPy (`cupyx.scipy.sparse.csr_matrix`).
   - Replace `A @ x` and related multiplies in CompressRTP with GPU equivalents.
   - Keep data on GPU for the full compression step.
2) **Compression steps**
   - Sparse-only: GPU sparse ops + thresholding on GPU.
   - Sparse+low-rank: use GPU randomized SVD (CuPy or cuML if available).
   - Wavelet: GPU wavelet if available (otherwise keep CPU and skip in `--gpu` mode).
3) **Dose reshaping**
   - `dose_1d -> dose_3d` mapping can be GPU-accelerated if it is pure indexing/mapping.

Deliverable: same outputs as CPU, with faster compression build time where GPU kernels are used; solver still runs on CPU.

Current implementation notes:
- `--gpu` flag + UI toggle added (optional, off by default).
- GPU is used for `dose_1d = A @ x` (CuPy), with CPU fallback on error.
- Sparse+low-rank thresholding can use GPU arrays with chunking; SVD remains CPU (same algorithm).
- RMR sparse-only + wavelet basis remain CPU (logged fallback).
- Chunk size for GPU compression is tunable via `ECHO_GPU_COMPRESS_CHUNK_MB`.

### Phase 2: Optional solver exploration (likely CPU)
- If GPU solver support is required, evaluate SCS/CVXOPT alternatives.
- Only proceed if solver matches the same formulation and accuracy thresholds.

### Phase 3: End-to-end run on GPU-accelerated kernels
- Validate a full CompressRTP run with GPU kernels + CPU solver:
  - Convergence behavior identical
  - DVH + metrics match tolerances
  - Peak GPU memory recorded

### Tests to require before enabling in UI
- `compressrtp sparse-only` with `--gpu`: matches CPU DVH/metrics (fast preset).
- `compressrtp sparse+low-rank` with `--gpu`: matches CPU DVH/metrics (fast preset).
- Stress test: ensure `--gpu` off reproduces prior results.

## Candidate GPU paths to evaluate
- Dose engine + influence matrix:
  - Prototype with CuPy for voxel-wise kernels and sparse matrix assembly.
  - Validate numerical parity vs. NumPy/SciPy outputs.
- Solver alternatives:
  - Check whether SCS or other conic solvers with GPU support are viable.
  - Evaluate CPU vs GPU tradeoffs with the same problem formulation.

## Minimal viability tests
- Convert one influence matrix build step to GPU and benchmark.
- Confirm identical DVH + metrics within tolerance.
- Measure peak GPU memory and transfer overhead.

## Open questions
- Is the PortPy dose engine structured for GPU kernels without major refactor?
- Can sparse matrix ops remain GPU-resident end-to-end?
- Will solver constraints allow GPU for the optimization stage?
