# TJU ECHO-VMAT Workbench

Local, research-only workbench for running ECHO-VMAT example patients, capturing plan quality (DVH + metrics), and timing the end-to-end optimization pipeline. The code wraps the official ECHO-VMAT example flow and avoids re-implementing solver logic.

## Current Status (Living Checklist)

- [x] Project scaffold created (`echo-workbench/` layout)
- [x] Python venv created (`echo-vmat-venv`, Python 3.10)
- [x] ECHO-VMAT cloned into `echo-workbench/echo-vmat`
- [x] ECHO-VMAT dependencies installed, MOSEK Python package installed
- [x] Runner wrapper (`backend/runner.py`) instrumented with timing + artifacts
- [x] Fast + super-fast modes for lower-resolution test runs
- [x] Adapter skeletons (example + Hugging Face)
- [x] FastAPI backend with run management + SSE logs + artifacts
- [x] Next.js UI with run setup, live logs, DVH, and clinical metrics
- [x] Standalone DVH + clinical criteria figure generator
- [~] Full-resolution example run validated (fast mode ok; full-resolution still pending)
- [ ] ESAPI adapter (future)

## Repository Layout

```
echo-workbench/
├── backend/                 # FastAPI app + runner
├── data/                    # dataset cache (raw/processed)
├── echo-vmat/               # ECHO-VMAT repo (submodule)
├── frontend/                # Next.js UI
└── README.md
```

## Quick Start

### 1) Submodule (ECHO-VMAT)
If you cloned this repo fresh:
```
git submodule update --init --recursive
```

### 2) Python venv
```
python -m venv echo-vmat-venv
source echo-vmat-venv/bin/activate
```

### 3) Install ECHO-VMAT requirements
```
python -m pip install -r echo-workbench/echo-vmat/requirements.txt
```

### 4) Run the ECHO example (CLI)
```
/mnt/d/_PROJECTS/ECHO-VMAT_Project/echo-vmat-venv/bin/python \
  /mnt/d/_PROJECTS/ECHO-VMAT_Project/echo-workbench/backend/runner.py \
  --case-id Lung_Patient_11 \
  --data-dir /mnt/d/_PROJECTS/ECHO-VMAT_Project/echo-workbench/PortPy/data \
  --super-fast
```

### 5) Backend API
```
cd /mnt/d/_PROJECTS/ECHO-VMAT_Project/echo-workbench
python -m uvicorn backend.main:app --reload --port 8000
```

### 6) Frontend UI
```
cd /mnt/d/_PROJECTS/ECHO-VMAT_Project/echo-workbench/frontend
npm install
npm run dev
```
Open: http://localhost:3000

## Artifacts Per Run
All run outputs are saved under:
```
echo-workbench/backend/runs/<run_id>/
```
Key files:
- `config.json`, `status.json`, `timing.json`
- `metrics.json` (clinical criteria table)
- `dvh.json` + `dvh_steps.png`
- `solver_trace.json`
- `clinical_criteria.html` (shareable report)

## Data Management
- Raw data is expected under `echo-workbench/PortPy/data`.
- Hugging Face datasets cache to `echo-workbench/data/raw/huggingface/`.
- Raw data is never modified.

## Notes / Known Issues
- Full-resolution runs can be long and memory-heavy; use `--fast` or `--super-fast` for smoke tests.
- Some datasets may be missing `voxel_coordinate_XYZ_mm`; the runner computes a fallback using PortPy utilities.
- Warnings about DVH dose limits can appear if dose thresholds exceed max dose; these do not stop the run.

## License
PortPy uses Apache 2.0 with a Commons Clause (non-commercial research only). Review upstream license terms before distribution.

