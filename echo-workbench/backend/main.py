import asyncio
import json
import threading
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

try:
    from backend.runner import _default_data_dir, run_echo_example
except ImportError:
    from runner import _default_data_dir, run_echo_example

try:
    from adapters.example_adapter import ExampleAdapter
    from adapters.huggingface_adapter import HuggingFaceAdapter
except ImportError:
    from backend.adapters.example_adapter import ExampleAdapter
    from backend.adapters.huggingface_adapter import HuggingFaceAdapter


RUNS_DIR = Path(__file__).resolve().parent / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


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
                    for line in handle:
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
