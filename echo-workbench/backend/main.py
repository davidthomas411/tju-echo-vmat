import asyncio
import io
import json
import logging
import re
import threading
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

import h5py
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

try:
    from backend.runner import _default_data_dir, run_echo_example, _ensure_echo_vmat_on_path
except ImportError:
    from runner import _default_data_dir, run_echo_example, _ensure_echo_vmat_on_path

try:
    from adapters.example_adapter import ExampleAdapter
    from adapters.huggingface_adapter import HuggingFaceAdapter
except ImportError:
    from backend.adapters.example_adapter import ExampleAdapter
    from backend.adapters.huggingface_adapter import HuggingFaceAdapter


RUNS_DIR = Path(__file__).resolve().parent / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURE_COLORS = [
    (96, 165, 250),
    (249, 115, 22),
    (34, 211, 238),
    (250, 204, 21),
    (52, 211, 153),
    (244, 114, 182),
    (167, 139, 250),
    (248, 113, 113),
    (56, 189, 248),
    (251, 113, 133),
    (74, 222, 128),
    (232, 121, 249),
]


class RunRequest(BaseModel):
    case_id: str
    protocol: str = "Lung_2Gy_30Fx"
    adapter: Literal["example", "huggingface"] = "example"
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    hf_subdir: Optional[str] = None
    use_planner_beams: bool = False
    use_available_beams: bool = False
    force_sparse: bool = False
    fast: bool = False
    super_fast: bool = False


class RunResponse(BaseModel):
    run_id: str


app = FastAPI()
access_logger = logging.getLogger("uvicorn.access")
access_logger.disabled = True
access_logger.propagate = False
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_run_config(run_id: str) -> dict:
    out_dir = RUNS_DIR / run_id
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="config.json not found for run")
    return _read_json(config_path)


def _resolve_case_dir(run_id: str) -> tuple[Path, dict]:
    config = _load_run_config(run_id)
    case_id = config.get("case_id")
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id missing in run config")
    data_dir = Path(config.get("data_dir") or _default_data_dir())
    direct = data_dir / case_id
    nested = data_dir / case_id / case_id
    if direct.is_dir():
        return direct, config
    if nested.is_dir():
        return nested, config
    raise HTTPException(status_code=404, detail="case data directory not found")


def _ct_reference(case_dir: Path) -> tuple[dict, Path, str]:
    meta_path = case_dir / "CT_MetaData.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="CT_MetaData.json not found")
    meta = _read_json(meta_path)
    file_ref = meta.get("ct_hu_3d_File", "CT_Data.h5/ct_hu_3d")
    if "/" not in file_ref:
        raise HTTPException(status_code=500, detail="Invalid ct_hu_3d_File reference")
    file_name, dataset = file_ref.split("/", 1)
    ct_path = case_dir / file_name
    if not ct_path.exists():
        raise HTTPException(status_code=404, detail="CT data file not found")
    return meta, ct_path, dataset


def _structure_entries(case_dir: Path) -> list[dict]:
    meta_path = case_dir / "StructureSet_MetaData.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="StructureSet_MetaData.json not found")
    meta = _read_json(meta_path)
    entries = []
    for item in meta:
        name = item.get("name")
        file_ref = item.get("structure_mask_3d_File")
        if not name or not file_ref or "/" not in file_ref:
            continue
        file_name, dataset = file_ref.split("/", 1)
        path = case_dir / file_name
        if not path.exists():
            continue
        entries.append({"name": name, "path": path, "dataset": dataset})
    if not entries:
        raise HTTPException(status_code=404, detail="No structure masks found")
    return entries


def _mask_edge(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    edge = mask & (
        (~padded[:-2, 1:-1])
        | (~padded[2:, 1:-1])
        | (~padded[1:-1, :-2])
        | (~padded[1:-1, 2:])
    )
    return edge


def _window_ct(slice_hu: np.ndarray, window: float, level: float) -> np.ndarray:
    if window <= 0:
        window = 1.0
    min_val = level - window / 2.0
    max_val = level + window / 2.0
    clipped = np.clip(slice_hu, min_val, max_val)
    scaled = (clipped - min_val) / (max_val - min_val)
    return (scaled * 255.0).astype(np.uint8)


def _colorize_dose(slice_gy: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (slice_gy - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    xp = np.array([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float32)
    r = np.interp(norm, xp, [0, 0, 41, 255, 255])
    g = np.interp(norm, xp, [0, 128, 220, 200, 64])
    b = np.interp(norm, xp, [0, 255, 255, 64, 0])
    alpha = np.where(norm > 0, (norm ** 0.7) * 255.0, 0.0)
    rgba = np.stack([r, g, b, alpha], axis=-1)
    return rgba.astype(np.uint8)


def _run_worker(run_id: str, req: RunRequest, out_dir: Path) -> None:
    adapter = None
    if req.adapter == "example":
        adapter = ExampleAdapter(data_root=_default_data_dir())
    elif req.adapter == "huggingface":
        if not req.hf_repo_id:
            _write_json(
                out_dir / "status.json",
                {"state": "error", "error": "hf_repo_id is required for huggingface adapter"},
            )
            return
        adapter = HuggingFaceAdapter(
            repo_id=req.hf_repo_id,
            cache_dir=Path(__file__).resolve().parents[1] / "data" / "raw" / "huggingface",
            token=req.hf_token,
            subdir=req.hf_subdir,
        )

    use_planner_beams = req.use_planner_beams or req.fast or req.super_fast
    use_available_beams = req.use_available_beams or req.fast or req.super_fast
    force_sparse = req.force_sparse or req.fast or req.super_fast

    try:
        run_echo_example(
            case_id=req.case_id,
            protocol_name=req.protocol,
            data_dir=_default_data_dir(),
            out_dir=out_dir,
            use_planner_beams=use_planner_beams,
            use_available_beams=use_available_beams,
            force_sparse=force_sparse,
            super_fast=req.super_fast,
            adapter=adapter,
        )
    except Exception as exc:
        _write_json(out_dir / "status.json", {"state": "error", "error": str(exc)})


@app.post("/runs", response_model=RunResponse)
def create_run(req: RunRequest) -> RunResponse:
    run_id = uuid4().hex
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "status.json", {"state": "queued"})

    thread = threading.Thread(target=_run_worker, args=(run_id, req, out_dir), daemon=True)
    thread.start()
    return RunResponse(run_id=run_id)


@app.get("/runs")
def list_runs() -> dict:
    runs = []
    for path in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        status_path = path / "status.json"
        status = _read_json(status_path) if status_path.exists() else {"state": "unknown"}
        runs.append({"run_id": path.name, "status": status})
    return {"runs": runs}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    out_dir = RUNS_DIR / run_id
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    status_path = out_dir / "status.json"
    status = _read_json(status_path) if status_path.exists() else {"state": "unknown"}
    artifacts = sorted([p.name for p in out_dir.iterdir() if p.is_file()])
    return {"run_id": run_id, "status": status, "artifacts": artifacts}


@app.get("/runs/{run_id}/events")
async def stream_events(run_id: str) -> StreamingResponse:
    out_dir = RUNS_DIR / run_id
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    events_path = out_dir / "events.jsonl"
    status_path = out_dir / "status.json"

    async def event_stream():
        last_pos = 0
        while True:
            if events_path.exists():
                with events_path.open("r", encoding="utf-8") as handle:
                    handle.seek(last_pos)
                    while True:
                        line = handle.readline()
                        if not line:
                            break
                        last_pos = handle.tell()
                        yield f"data: {line.strip()}\n\n"
            if status_path.exists():
                status = _read_json(status_path)
                if status.get("state") in {"completed", "error"}:
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/runs/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str) -> FileResponse:
    out_dir = RUNS_DIR / run_id
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    if Path(name).name != name:
        raise HTTPException(status_code=400, detail="invalid artifact name")
    path = out_dir / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path)


@app.get("/runs/{run_id}/ct/meta")
def get_ct_meta(run_id: str) -> dict:
    case_dir, _config = _resolve_case_dir(run_id)
    meta, ct_path, dataset = _ct_reference(case_dir)
    with h5py.File(ct_path, "r") as handle:
        shape = list(handle[dataset].shape)
    return {
        "shape_zyx": shape,
        "slice_count": shape[0] if shape else 0,
        "origin_xyz_mm": meta.get("origin_xyz_mm"),
        "resolution_xyz_mm": meta.get("resolution_xyz_mm"),
        "direction": meta.get("direction"),
    }


@app.get("/runs/{run_id}/ct/slice")
def get_ct_slice(
    run_id: str,
    index: int = Query(0, ge=0),
    window: float = Query(400.0, gt=0),
    level: float = Query(40.0),
) -> Response:
    case_dir, _config = _resolve_case_dir(run_id)
    _meta, ct_path, dataset = _ct_reference(case_dir)
    with h5py.File(ct_path, "r") as handle:
        ct = handle[dataset]
        max_index = ct.shape[0] - 1
        slice_index = min(index, max_index)
        slice_hu = ct[slice_index, :, :]
    slice_u8 = _window_ct(slice_hu, window=window, level=level)
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pillow not installed: {exc}") from exc
    image = Image.fromarray(slice_u8, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/runs/{run_id}/structures")
def list_structures(run_id: str) -> dict:
    case_dir, _config = _resolve_case_dir(run_id)
    entries = _structure_entries(case_dir)
    return {"structures": [entry["name"] for entry in entries]}


@app.get("/runs/{run_id}/structures/slice")
def get_structure_slice(
    run_id: str,
    index: int = Query(0, ge=0),
    names: Optional[str] = Query(None),
) -> Response:
    case_dir, _config = _resolve_case_dir(run_id)
    entries = _structure_entries(case_dir)
    name_filter = None
    if names:
        name_filter = {name.strip().upper() for name in names.split(",") if name.strip()}
    selected = [
        entry for entry in entries if name_filter is None or entry["name"].upper() in name_filter
    ]
    if not selected:
        raise HTTPException(status_code=404, detail="No matching structures found")
    overlay = None
    color_idx = 0
    entries_by_path = {}
    for entry in selected:
        entries_by_path.setdefault(entry["path"], []).append(entry)
    for path, path_entries in entries_by_path.items():
        with h5py.File(path, "r") as handle:
            for entry in path_entries:
                data = handle[entry["dataset"]]
                max_index = data.shape[0] - 1
                slice_index = min(index, max_index)
                mask_slice = data[slice_index, :, :]
                mask = mask_slice > 0
                edge = _mask_edge(mask)
                if overlay is None:
                    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                color = STRUCTURE_COLORS[color_idx % len(STRUCTURE_COLORS)]
                overlay[edge, 0] = color[0]
                overlay[edge, 1] = color[1]
                overlay[edge, 2] = color[2]
                overlay[edge, 3] = 200
                color_idx += 1
    if overlay is None:
        raise HTTPException(status_code=404, detail="No structure overlay generated")
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pillow not installed: {exc}") from exc
    image = Image.fromarray(overlay, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/runs/{run_id}/dose-3d")
def create_dose_3d(run_id: str, overwrite: bool = Query(False)) -> dict:
    out_dir = RUNS_DIR / run_id
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    dose_path = out_dir / "dose_3d.npy"
    meta_path = out_dir / "dose_3d_meta.json"
    if dose_path.exists() and not overwrite:
        return {"status": "exists", "artifact": dose_path.name}
    try:
        import portpy.photon as pp
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"portpy not available: {exc}") from exc
    plan_path = out_dir / "my_plan.pkl"
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="my_plan.pkl not found")
    sol_paths = list(out_dir.glob("sol_step*.pkl"))
    if not sol_paths:
        raise HTTPException(status_code=404, detail="No sol_step*.pkl found for run")
    sol_paths.sort(
        key=lambda path: int(re.search(r"sol_step(\\d+)", path.stem).group(1))
        if re.search(r"sol_step(\\d+)", path.stem)
        else -1
    )
    sol_path = sol_paths[-1]
    _ensure_echo_vmat_on_path()
    my_plan = pp.load_plan(plan_name=plan_path.name, path=str(out_dir))
    sol = pp.load_optimal_sol(sol_name=sol_path.name, path=str(out_dir))
    if "act_dose_v" in sol:
        dose_1d = sol["act_dose_v"] * my_plan.get_num_of_fractions()
    else:
        intensity = sol.get("optimal_intensity")
        if intensity is None:
            raise HTTPException(status_code=500, detail="Solution missing act_dose_v and optimal_intensity")
        dose_1d = my_plan.inf_matrix.A @ intensity * my_plan.get_num_of_fractions()
    dose_3d = my_plan.inf_matrix.dose_1d_to_3d(dose_1d=dose_1d).astype(np.float32)
    np.save(dose_path, dose_3d, allow_pickle=False)
    dose_min = float(np.min(dose_3d))
    dose_max = float(np.max(dose_3d))
    case_dir, config = _resolve_case_dir(run_id)
    meta, _ct_path, _dataset = _ct_reference(case_dir)
    _write_json(
        meta_path,
        {
            "run_id": run_id,
            "case_id": config.get("case_id"),
            "shape_zyx": list(dose_3d.shape),
            "origin_xyz_mm": meta.get("origin_xyz_mm"),
            "resolution_xyz_mm": meta.get("resolution_xyz_mm"),
            "direction": meta.get("direction"),
            "units": "Gy",
            "source_solution": sol_path.name,
            "min_gy": dose_min,
            "max_gy": dose_max,
        },
    )
    return {"status": "created", "artifact": dose_path.name}


@app.get("/runs/{run_id}/dose/slice")
def get_dose_slice(
    run_id: str,
    index: int = Query(0, ge=0),
    dose_min: float = Query(0.0, ge=0.0),
    dose_max: Optional[float] = Query(None),
) -> Response:
    out_dir = RUNS_DIR / run_id
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    dose_path = out_dir / "dose_3d.npy"
    if not dose_path.exists():
        raise HTTPException(status_code=404, detail="dose_3d.npy not found")
    meta_path = out_dir / "dose_3d_meta.json"
    if dose_max is None and meta_path.exists():
        meta = _read_json(meta_path)
        dose_max = meta.get("max_gy")
    dose_arr = np.load(dose_path, mmap_mode="r")
    max_index = dose_arr.shape[0] - 1
    slice_index = min(index, max_index)
    slice_gy = dose_arr[slice_index, :, :]
    if dose_max is None:
        dose_max = float(np.max(slice_gy))
    rgba = _colorize_dose(slice_gy, vmin=dose_min, vmax=dose_max)
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Pillow not installed: {exc}") from exc
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})
