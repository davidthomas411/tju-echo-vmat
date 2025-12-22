# Optimization Profiling Notes

Goal: capture peak RAM + wall-clock timing for full-resolution runs and document best-fit presets.

## Where the numbers come from
- Each run writes `timing.json` with stage durations and `max_rss_mb` (peak resident set size).
- The UI shows these values in the Timing + Resource Summary panels.

## Suggested profiling workflow
1) Start with a safe preset to validate the pipeline:
```
python echo-workbench/backend/runner.py \
  --case-id Lung_Patient_11 \
  --data-dir /mnt/d/_PROJECTS/ECHO-VMAT_Project/echo-workbench/PortPy/data \
  --super-fast
```

2) Move to a full-resolution run:
```
python echo-workbench/backend/runner.py \
  --case-id Lung_Patient_11 \
  --data-dir /mnt/d/_PROJECTS/ECHO-VMAT_Project/echo-workbench/PortPy/data
```

3) Record `timing.json` + `status.json`:
- `timing.json` → stage timings + peak RAM
- `status.json` → elapsed time + last RSS sample

## Preset comparisons (fill in per patient)
- Super-fast: ________ sec, peak RAM ________ MB
- Fast: ________ sec, peak RAM ________ MB
- Full: ________ sec, peak RAM ________ MB

## Next tighten-ups to evaluate
- Sparse influence matrix (DDC) where possible
- Fewer beams for compressrtp test runs
- Keep dose export optional to reduce memory pressure
