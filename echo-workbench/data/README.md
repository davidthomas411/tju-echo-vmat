# Data Sources

This folder contains raw and processed datasets used by the ECHO-VMAT workbench.

## Raw
- `raw/huggingface/`: Hugging Face dataset cache (downloaded on demand).

## Processed
- `processed/`: Any normalized or derived data created for runs.
  - `processed/dicom/<case_id>/ct/`: Generated CT DICOM series (created once per patient).
  - `processed/dicom/<case_id>/rtstruct/`: Generated RT Structure Set DICOM (created once per patient).

## Provenance
Record dataset name, version, and download date for reproducibility.
