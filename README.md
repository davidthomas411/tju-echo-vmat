# TJU ECHO-VMAT Workbench


![ECHO-VMAT Workbench UI](docs/screenshots/ui-2025-12-22.jpg)


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
- [x] Interactive DVH plot (hover values, percent axes)
- [x] Run comparison (overlay DVHs + metric deltas)
- [x] CT viewer with window/level + wheel slice navigation
- [x] Structure overlay (per-slice outlines)
- [x] Optional 3D dose export + CT/dose color overlay
- [x] RT Plan DICOM export (from ECHO template plan)
- [x] RT Dose DICOM export (per run)
- [x] CT DICOM export (per patient, generated once)
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
Optional (RT Plan export):
```
python -m pip install "portpy[pydicom]"
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

UI notes:
- Load a run from the Run Queue.
- Click "Create 3D Dose" once to save `dose_3d.npy` for that run.
- Toggle Dose Overlay in the CT viewer (fast, no recompute).
- Use Run Comparison to overlay two DVHs and compute metric deltas.
- Use "Generate RT Plan" to export DICOM RT Plan + RT Dose for TPS import.


CT DICOM export (once per patient):
```
curl -X POST http://127.0.0.1:8000/runs/<run_id>/ct-dicom
```
RT Structure Set export (once per patient):
```
curl -X POST http://127.0.0.1:8000/runs/<run_id>/rtstruct
```

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
- `dose_3d.npy` + `dose_3d_meta.json` (optional, generated on demand)
- `rt_plan_portpy_vmat.dcm` (optional, generated on demand)
- `rt_dose_portpy_vmat.dcm` (optional, generated on demand)

CT DICOM (generated once per patient) is saved under:
```
echo-workbench/data/processed/dicom/<case_id>/ct/
```
RT Structure Set DICOM (generated once per patient) is saved under:
```
echo-workbench/data/processed/dicom/<case_id>/rtstruct/rt_struct_portpy.dcm
```

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
