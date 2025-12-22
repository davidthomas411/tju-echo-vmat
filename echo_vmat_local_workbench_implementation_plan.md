# ECHO‑VMAT Local Workbench – Implementation Plan

**Goal (24‑hour MVP)**  
By end of day, stand up a **local, modern UI application** that runs **ECHO‑VMAT example patients** from the official GitHub repo, reports **plan quality** and **runtime breakdown**, and is architected to later plug into **Eclipse ESAPI** with minimal refactor.

This is a **research / pre‑clinical workbench**, not a clinical system.

---

## 1. Scope and Non‑Goals

### In scope (today)
- Run ECHO‑VMAT **example cases** exactly as implemented in the repo
- Measure:
  - Wall‑clock runtime (DDC, optimization, correction loop, evaluation)
  - Plan quality (DVH + key metrics)
- Interactive **local web UI** (localhost)
- TJU‑branded, modern, fast UI
- Clean separation between:
  - data loading
  - optimization
  - dose recomputation
  - evaluation

### Explicitly out of scope (today)
- Eclipse / ESAPI integration
- FDA / clinical workflow compliance
- Re‑implementing or modifying ECHO algorithms
- Proprietary Varian / MSK DDC engines

---

## 2. Guiding Principles

1. **Follow the ECHO‑VMAT example code as closely as possible**  
   Wrap it — do not rewrite it.

2. **Instrument, don’t optimize**  
   Today’s goal is *measurement* (quality + time), not speed.

3. **Future‑proof interfaces**  
   Design now for ESAPI integration later.

4. **Deterministic, inspectable runs**  
   Every run produces a complete artifact folder.

---

## 2.1 Current Status (Living)

- [x] Created `echo-workbench/` scaffold and required subfolders
- [x] Created Python venv `echo-vmat-venv` (Python 3.10)
- [x] Cloned official ECHO-VMAT repo into `echo-workbench/echo-vmat`
- [x] Installed ECHO-VMAT dependencies + MOSEK Python package
- [x] Verified example run (fast/super-fast) produces artifacts end-to-end
- [~] Full-resolution example run validated (still pending)
- [x] Added `backend/runner.py` wrapper with voxel-coordinate fallback and fast/super-fast modes
- [x] Fixed evaluation call to `get_low_dose_vox_ind` and hardened metrics export (Styler -> DataFrame)
- [x] Implemented adapter skeletons and FastAPI backend endpoints (runs + SSE + artifacts)
- [x] Added initial Next.js frontend (run setup, live progress, DVH/metrics/timing views)
- [x] UI now supports loading existing runs and shows DVH image + metrics table
- [x] Runner now writes a shareable `clinical_criteria.html` artifact per run
- [x] UI refreshed to an enterprise-style console layout with live trace chart + run queue
- [x] Interactive DVH plot (hover values, percent axes, focus filter)
- [x] Run comparison (overlay DVHs + metric delta table)
- [x] CT viewer with window/level + wheel navigation (square viewport)
- [x] Structure overlay (CT axial outlines)
- [x] Optional 3D dose export + CT/dose overlay (no recompute on display)
- [x] RT Plan DICOM export button (uses ECHO template plan)
- [x] Added CompressRTP submodule + runner integration (separate run root)
- [x] UI optimizer selector with CompressRTP modes (sparse-only, sparse+low-rank, wavelet)
- [x] CompressRTP artifacts (metrics/DVH/plan/solution) saved under `runs-compressrtp/`
- [x] CompressRTP integration tests (3 modes) pass on Lung_Patient_11
- [x] CompressRTP step diagnostics in UI (DDC-only, sparse, svd, wavelet)
- [x] Run tagging supported in backend + UI run lists
- [x] Stage-aware status + elapsed time + RSS sampling for profiling
- [x] DICOM exports panel (CT/RTSTRUCT + RTPLAN/RTDOSE) with deterministic paths

---

## 3. High‑Level Architecture

```
┌──────────────────────┐
│  Next.js UI (local)  │  ← TJU‑branded, interactive
└─────────┬────────────┘
          │ HTTP / SSE
┌─────────▼────────────┐
│ FastAPI Backend      │  ← thin orchestration layer
│  - run manager       │
│  - SSE logs          │
│  - artifact serving  │
└─────────┬────────────┘
          │ Python API
┌─────────▼────────────┐
│ ECHO Runner Wrapper  │  ← wraps repo examples
│  - timing hooks      │
│  - artifact capture  │
└─────────┬────────────┘
          │
┌─────────▼────────────┐
│ ECHO‑VMAT Repo Code  │  ← unmodified algorithms
└──────────────────────┘
```

---

## 4. Repository Layout

```
echo-workbench/
├── data/                     # downloaded datasets (HuggingFace cache)
│   ├── raw/                  # untouched datasets as provided
│   │   └── huggingface/
│   │       └── <dataset_name>/
│   ├── processed/            # normalized / preprocessed for ECHO
│   │   └── <case_id>/
│   └── README.md              # dataset provenance + versioning notes
│
├── backend/
│   ├── main.py              # FastAPI app
│   ├── runner.py            # ECHO wrapper
│   ├── runs/
│   │   └── <run_id>/
│   │       ├── config.json
│   │       ├── status.json
│   │       ├── timing.json
│   │       ├── metrics.json
│   │       ├── dvh.json
│   │       ├── clinical_criteria.json
│   │       ├── solver_trace.json
│   │       └── logs.txt
│   ├── runs-compressrtp/
│   │   └── <run_id>/
│   ├── runs-compressrtp-tests/
│   ├── adapters/
│   │   ├── base_adapter.py
│   │   ├── example_adapter.py
│   │   └── huggingface_adapter.py
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── styles/
│   └── theme/
│
├── echo-vmat/                # Git submodule or clone
└── README.md
```

echo-workbench/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── runner.py            # ECHO wrapper
│   ├── runs/
│   │   └── <run_id>/
│   │       ├── config.json
│   │       ├── status.json
│   │       ├── timing.json
│   │       ├── metrics.json
│   │       ├── dvh.json
│   │       ├── clinical_criteria.json
│   │       ├── solver_trace.json
│   │       └── logs.txt
│   └── adapters/
│       ├── example_adapter.py
│       └── base_adapter.py
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── styles/
│   └── theme/
│
├── echo-vmat/                # Git submodule or clone
└── README.md
```

---

## 5. ECHO Runner Wrapper (Critical Path)

### Purpose
Provide a **stable, minimal interface** around the existing ECHO-VMAT example workflow.

### Interface
```python
run_echo(
    case_id: str,
    preset: str,
    params: dict,
    out_dir: Path
) -> None
```

### Responsibilities
- Resolve dataset location (local cache or HuggingFace download)
- Load example case exactly as in repo
- Execute ECHO optimization (sequential convex programming)
- Execute dose correction loop (if present)
- Evaluate plan quality using repo utilities
- Capture all outputs + logs
- Record precise timing checkpoints

### Timing instrumentation (required)
- t_case_load
- t_dataset_download (if applicable)
- t_ddc_load_or_build
- t_optimization_total
- t_correction_loop_total
- t_evaluation
- t_total

---

## 6. Backend API (FastAPI)

### Endpoints
- `POST /runs`
  - input: `{ case_id, preset, params }`
  - output: `{ run_id }`

- `GET /runs/{run_id}`
  - status, progress, artifact index

- `GET /runs/{run_id}/events`
  - Server‑Sent Events (live logs + timing updates)

- `GET /runs/{run_id}/artifacts/{name}`
  - download JSON / CSV artifacts

### Execution model
- Background task per run
- File‑based state (no DB today)
- Deterministic, restartable

---

## 7. Frontend UI (Next.js)

### Design goals
- Clean, clinical, modern
- Fast interactions
- Minimal cognitive load
- TJU branding (header, colors, typography)

### Pages / Components

#### 7.1 Run Setup
- Example case selector
- Solver preset selector:
  - Fast
  - Balanced
  - High Quality
- Run button

#### 7.2 Live Run View
- Progress timeline:
  - Case load
  - DDC
  - SCP iterations
  - Correction loop
  - Evaluation
- Live log stream (SSE)
- Wall‑clock timers

#### 7.3 Results View
- DVH plot
- Key metrics table
- Clinical criteria pass/fail
- Runtime breakdown chart
- Artifact download links

---

## 8. Quality Metrics (MVP)

At minimum:
- Target coverage (D95, V100)
- OAR max / mean dose (site‑dependent)
- DVH visualization
- Pass/fail list aligned with example constraints

Do **not** invent new metrics — reuse repo outputs where possible.

---

## 9. Presets (Configuration Discipline)

Presets are **named parameter bundles**, not new logic.

Example:
```json
{
  "preset": "balanced",
  "scp_max_iters": 20,
  "correction_iters": 3,
  "tolerance": 1e-3
}
```

This mirrors how ECHO is tuned clinically without exposing raw solver internals in the UI.

---

## 10. Data Management & HuggingFace Integration

### Goals
- Reproducible datasets
- Explicit provenance
- Zero manual data handling

### Dataset source
- Example datasets hosted on **HuggingFace** (official / PortPy-aligned)
- Downloaded automatically on first use

### HuggingFace adapter
`huggingface_adapter.py` responsibilities:
- Check local cache under `data/raw/huggingface/`
- Download dataset via `datasets` or `huggingface_hub`
- Verify expected files / checksums if provided
- Expose normalized paths to ECHO runner

### Caching rules
- Never re-download if present
- Dataset version pinned in `data/README.md`
- All runs reference dataset by `{name}:{version}`

### Processed data
- Any normalization / reformatting for ECHO stored under:
  - `data/processed/<case_id>/`
- Raw data remains untouched

---

## 11. ESAPI Expansion Strategy (Design Now)

### Adapter pattern
```python
class CaseAdapter:
    def load_case(self, ref): ...
    def get_ddc(self): ...
    def export_plan(self, solution): ...
```

- Today: `ExampleAdapter`, `HuggingFaceAdapter`
- Future: `EclipseAdapter (ESAPI)`

### Dose engine abstraction
- Today: repo dose + correction loop
- Future: Eclipse final dose calculation

No UI changes required when swapping adapters.

---

## 12. Validation Checklist (End of Day)

- [ ] At least one ECHO‑VMAT example runs end‑to‑end
- [ ] Runtime breakdown visible in UI
- [ ] DVH + metrics displayed
- [ ] Artifacts saved per run
- [ ] UI responsive and branded
- [ ] Code paths clearly separable for ESAPI

---

## 13. Definition of Success

By end of day, you can:
- Launch a local web app
- Select an ECHO example patient
- Run optimization
- Watch progress live
- Compare plan quality and runtime
- Export artifacts for analysis

This establishes a **credible, extensible foundation** for future Eclipse‑integrated ECHO‑VMAT research and clinical translation.

---

## 14. Big Project List (Backlog)

- [~] Optimization: profile full‑resolution runs (peak RAM + time), tighten sparse DDC usage, and document best‑fit presets
  - Added `max_rss_mb` sampling + UI resource summary
  - Added `docs/optimization_profiling.md` checklist
- [~] GPU feasibility study: dose engine + influence matrix on GPU; assess solver alternatives for conic optimization
  - Added `docs/gpu_feasibility.md` with GPU plan + tests (CompressRTP-focused)
- [~] ESAPI adapter (Eclipse case + dose export)
  - Stub `EsapiAdapter` added; integration still pending
- [~] Multi‑run orchestration (batch queue + compare dashboard)
  - Added backend batch endpoint (`/runs/batch` + status)
