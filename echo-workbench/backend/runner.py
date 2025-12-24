import argparse
import os
import json
import re
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import portpy.photon as pp

matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from backend.gpu import estimate_chunk_rows, get_gpu_context, gpu_matmul
except ImportError:  # pragma: no cover - direct runner invocation
    from gpu import estimate_chunk_rows, get_gpu_context, gpu_matmul


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_data_dir() -> Path:
    return _repo_root() / "PortPy" / "data"


def _runs_root(run_type: str = "echo-vmat") -> Path:
    if run_type == "compressrtp":
        return _repo_root() / "backend" / "runs-compressrtp"
    return _repo_root() / "backend" / "runs"


def _default_out_dir(case_id: str, run_type: str = "echo-vmat") -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    root = _runs_root(run_type)
    if run_type == "compressrtp":
        return root / f"compressrtp-{case_id}-{ts}"
    return root / f"{case_id}-{ts}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def _log_event(out_dir: Path, stage: str, message: str, level: str = "info", data: dict | None = None) -> None:
    event = {
        "ts": _now_iso(),
        "stage": stage,
        "level": level,
        "message": message,
    }
    if data is not None:
        event["data"] = data
    _append_jsonl(out_dir / "events.jsonl", event)
    with (out_dir / "logs.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"[{event['ts']}] {level.upper()} {stage}: {message}\n")


def _rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss_kb <= 0:
        return None
    return rss_kb / 1024.0


def _start_heartbeat(
    out_dir: Path,
    stage: str,
    message_prefix: str,
    interval_sec: float = 60.0,
) -> threading.Event:
    stop_event = threading.Event()

    def _run() -> None:
        start = time.perf_counter()
        while not stop_event.wait(interval_sec):
            elapsed = time.perf_counter() - start
            _log_event(out_dir, stage, f"{message_prefix} ({elapsed:.0f}s elapsed)")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return stop_event


def _emit_trace(out_dir: Path, sol_convergence: list[dict], step_label: str) -> None:
    for sol in sol_convergence:
        data = {
            "step": step_label,
            "outer_iteration": sol.get("outer_iteration"),
            "inner_iteration": sol.get("inner_iteration"),
            "intermediate_obj_value": sol.get("intermediate_obj_value"),
            "actual_obj_value": sol.get("actual_obj_value"),
            "accept": sol.get("accept"),
        }
        _log_event(out_dir, "trace", "trace_point", data=data)


def _scale_weight(weight: object, scale: float) -> object:
    if weight is None or scale == 1.0:
        return weight
    if isinstance(weight, list):
        return [float(val) * scale for val in weight]
    try:
        return float(weight) * scale
    except (TypeError, ValueError):
        return weight


def _apply_objective_weight_scale(objectives: list[dict], scale_map: dict) -> None:
    if not objectives or not scale_map:
        return
    target_scale = float(scale_map.get("target", 1.0))
    oar_scale = float(scale_map.get("oar", 1.0))
    aperture_scale = float(scale_map.get("aperture", 1.0))
    dfo_scale = float(scale_map.get("dfo", 1.0))
    for obj in objectives:
        weight = obj.get("weight")
        if weight is None:
            continue
        obj_type = str(obj.get("type", "")).lower()
        if obj_type.startswith("aperture_"):
            scale = aperture_scale
        elif obj_type == "dfo":
            scale = dfo_scale
        elif obj.get("is_target", 0):
            scale = target_scale
        else:
            scale = oar_scale
        if scale != 1.0:
            obj["weight"] = _scale_weight(weight, scale)


def _objective_matches(obj: dict, override: dict) -> bool:
    obj_type = str(obj.get("type", "")).lower()
    if "type" in override and str(override.get("type", "")).lower() != obj_type:
        return False
    if "structure_name" in override:
        if str(override.get("structure_name", "")).upper() != str(obj.get("structure_name", "")).upper():
            return False
    if "structure_def" in override:
        if str(override.get("structure_def", "")).upper() != str(obj.get("structure_def", "")).upper():
            return False
    return True


def _apply_explicit_objective_weights(objectives: list[dict], overrides: list[dict]) -> None:
    if not objectives or not overrides:
        return
    for override in overrides:
        if "weight" not in override:
            continue
        for obj in objectives:
            if _objective_matches(obj, override):
                obj["weight"] = override["weight"]


def _filter_step_overrides(overrides: list[dict], step_id: str | None) -> list[dict]:
    if not overrides:
        return []
    filtered: list[dict] = []
    for override in overrides:
        step = override.get("step")
        if step is None:
            filtered.append(override)
        elif step_id is not None and str(step) == str(step_id):
            filtered.append(override)
    return filtered


def _apply_opt_params_overrides(opt_params: dict, overrides: dict | None) -> dict:
    if not overrides or not isinstance(overrides, dict):
        return opt_params
    updated = deepcopy(opt_params)
    opt_overrides = overrides.get("opt_parameters")
    if isinstance(opt_overrides, dict):
        updated.setdefault("opt_parameters", {}).update(opt_overrides)
    scale_map = overrides.get("objective_weight_scale")
    weight_overrides = overrides.get("objective_weights")
    if "steps" in updated:
        for step_id, step_cfg in updated.get("steps", {}).items():
            objectives = step_cfg.get("objective_functions", [])
            if isinstance(scale_map, dict):
                _apply_objective_weight_scale(objectives, scale_map)
            if isinstance(weight_overrides, list):
                _apply_explicit_objective_weights(
                    objectives, _filter_step_overrides(weight_overrides, str(step_id))
                )
    else:
        objectives = updated.get("objective_functions", [])
        if isinstance(scale_map, dict):
            _apply_objective_weight_scale(objectives, scale_map)
        if isinstance(weight_overrides, list):
            _apply_explicit_objective_weights(objectives, _filter_step_overrides(weight_overrides, None))
    return updated


def _parse_opt_params_overrides(raw: str | None) -> dict | None:
    if not raw:
        return None
    if raw.startswith("@"):
        path = Path(raw[1:])
        if not path.exists():
            raise FileNotFoundError(f"Overrides file not found: {path}")
        return _read_json(path)
    return json.loads(raw)


def _update_status(out_dir: Path, payload: dict) -> None:
    _write_json(out_dir / "status.json", payload)


def _merge_status(out_dir: Path, payload: dict) -> None:
    status_path = out_dir / "status.json"
    status = _read_json(status_path) if status_path.exists() else {}
    status.update(payload)
    _write_json(status_path, status)


def _normalize_data_dir(data_dir: Path, case_id: str) -> Path:
    if (data_dir / case_id).is_dir():
        return data_dir
    if (data_dir / case_id / case_id).is_dir():
        return data_dir / case_id
    return data_dir


def _load_planner_beam_ids(patient_dir: Path) -> list[int] | None:
    planner_file = patient_dir / "PlannerBeams.json"
    if not planner_file.exists():
        return None
    with planner_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    ids = payload.get("IDs")
    if not ids:
        return None
    return sorted({int(v) for v in ids})


def _available_beam_ids(patient_dir: Path) -> list[int] | None:
    beams_dir = patient_dir / "Beams"
    if not beams_dir.exists():
        return None
    ids: list[int] = []
    for meta_path in beams_dir.glob("Beam_*_MetaData.json"):
        match = re.search(r"Beam_(\d+)_MetaData\.json", meta_path.name)
        if match:
            ids.append(int(match.group(1)))
    if not ids:
        return None
    return sorted(set(ids))


def _parse_beam_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    parts = re.split(r"[,\s]+", raw.strip())
    ids = [int(part) for part in parts if part]
    return ids or None


def _matrix_info(matrix) -> dict:
    shape = list(matrix.shape) if hasattr(matrix, "shape") else None
    nnz = getattr(matrix, "nnz", None)
    return {
        "shape": shape,
        "nnz": nnz,
        "sparse": nnz is not None,
    }


def _ensure_echo_vmat_on_path() -> None:
    echo_vmat_root = _repo_root() / "echo-vmat"
    if str(echo_vmat_root) not in sys.path:
        sys.path.insert(0, str(echo_vmat_root))


def _ensure_compressrtp_on_path() -> None:
    compress_root = _repo_root() / "compressrtp"
    if str(compress_root) not in sys.path:
        sys.path.insert(0, str(compress_root))


def _ensure_voxel_coordinates(inf_matrix: pp.InfluenceMatrix) -> None:
    if "voxel_coordinate_XYZ_mm" not in inf_matrix.opt_voxels_dict:
        coords = inf_matrix.get_voxel_coordinates()
        inf_matrix.opt_voxels_dict["voxel_coordinate_XYZ_mm"] = [coords]


def _select_struct_names(my_plan: pp.Plan) -> list[str]:
    preferred = [
        "PTV",
        "ESOPHAGUS",
        "HEART",
        "CORD",
        "RIND_0",
        "RIND_1",
        "LUNGS_NOT_GTV",
        "RECT_WALL",
        "BLAD_WALL",
        "URETHRA",
        "LUNG_L",
        "LUNG_R",
        "SKIN",
        "BODY",
    ]
    available = my_plan.structures.structures_dict["name"]
    selected = [name for name in preferred if name in available]
    return selected if selected else available


def run_echo_example(
    case_id: str = "Lung_Phantom_Patient_1",
    protocol_name: str = "Lung_2Gy_30Fx",
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    solver: str = "MOSEK",
    use_planner_beams: bool = False,
    use_available_beams: bool = False,
    force_sparse: bool = False,
    super_fast: bool = False,
    opt_params_overrides: dict | None = None,
    tag: str | None = None,
    adapter=None,
) -> Path:
    _ensure_echo_vmat_on_path()
    from echo_vmat.arcs import Arcs
    from echo_vmat.echo_vmat_optimization import EchoVmatOptimization
    from echo_vmat.echo_vmat_optimization_col_gen import EchoVmatOptimizationColGen
    from echo_vmat.utils.get_sparse_only import get_sparse_only
    from portpy.photon.evaluation import Evaluation
    try:
        from backend.adapters.example_adapter import ExampleAdapter
    except ImportError:
        from adapters.example_adapter import ExampleAdapter

    data_dir = _normalize_data_dir(data_dir or _default_data_dir(), case_id)
    out_dir = out_dir or _default_out_dir(case_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_dir = data_dir / case_id

    if adapter is None:
        adapter = ExampleAdapter(data_root=data_dir)
    case_info = adapter.prepare_case(case_id)
    data_dir = _normalize_data_dir(case_info.data_dir, case_id)
    patient_dir = data_dir / case_id

    run_config = {
        "case_id": case_id,
        "protocol": protocol_name,
        "solver": solver,
        "data_dir": str(data_dir),
        "optimizer": "echo-vmat",
        "use_planner_beams": use_planner_beams,
        "use_available_beams": use_available_beams,
        "force_sparse": force_sparse,
        "super_fast": super_fast,
        "adapter": case_info.source,
        "dataset_download_sec": float(case_info.download_seconds),
    }
    if tag:
        run_config["tag"] = tag
    if opt_params_overrides:
        run_config["opt_params_overrides"] = opt_params_overrides
    _write_json(out_dir / "config.json", run_config)
    started_at = _now_iso()
    _update_status(
        out_dir,
        {
            "state": "running",
            "started_at": started_at,
            "case_id": case_id,
            "protocol": protocol_name,
        },
    )
    _log_event(out_dir, "start", "Run started")

    run_start = time.perf_counter()
    stage_starts: dict[str, float] = {}
    rss_samples: list[float] = []

    def _stage_begin(stage: str, message: str | None = None) -> None:
        stage_starts[stage] = time.perf_counter()
        elapsed = stage_starts[stage] - run_start
        rss_mb = _rss_mb()
        if rss_mb is not None:
            rss_samples.append(rss_mb)
        _merge_status(out_dir, {"stage": stage, "elapsed_sec": elapsed, "rss_mb": rss_mb})
        _log_event(
            out_dir,
            stage,
            message or f"{stage} started",
            data={"state": "start", "elapsed_sec": elapsed, "rss_mb": rss_mb},
        )

    def _stage_end(stage: str, message: str, extra: dict | None = None) -> float:
        end = time.perf_counter()
        start = stage_starts.get(stage, end)
        elapsed = end - run_start
        duration = end - start
        rss_mb = _rss_mb()
        if rss_mb is not None:
            rss_samples.append(rss_mb)
        data = {"state": "done", "elapsed_sec": elapsed, "duration_sec": duration, "rss_mb": rss_mb}
        if extra:
            data.update(extra)
        _log_event(out_dir, stage, message, data=data)
        return duration

    # Match the official example flow as closely as possible.
    tic_all = time.time()
    t_total_start = time.perf_counter()
    timing: dict[str, float | dict] = {
        "dataset_download_sec": float(case_info.download_seconds),
    }
    if case_info.download_seconds:
        _log_event(out_dir, "dataset_download", f"Downloaded dataset in {case_info.download_seconds:.2f}s")
    else:
        _log_event(
            out_dir,
            "dataset_download",
            "Dataset already available",
            data={"state": "done", "duration_sec": 0.0},
        )
    _stage_begin("case_load", "Loading case data")
    t_case_start = time.perf_counter()
    data = pp.DataExplorer(data_dir=str(data_dir))
    data.patient_id = case_id

    config_path = _repo_root() / "echo-vmat" / "echo_vmat" / "config_files"
    opt_params_path = config_path / f"{protocol_name}_opt_params.json"
    criteria_path = config_path / f"{protocol_name}_clinical_criteria.json"
    vmat_opt_params = data.load_json(file_name=str(opt_params_path))
    clinical_criteria = pp.ClinicalCriteria(file_name=str(criteria_path))

    if super_fast:
        use_planner_beams = True
        use_available_beams = True
        vmat_opt_params.setdefault("opt_parameters", {})["initial_leaf_pos"] = "BEV"
        vmat_opt_params["opt_parameters"]["min_iteration_threshold"] = 3
        vmat_opt_params["opt_parameters"]["termination_gap"] = 5
        vmat_opt_params["opt_parameters"]["step_size_increment"] = 2
        vmat_opt_params["opt_parameters"]["max_iteration_corr"] = 2
        vmat_opt_params["opt_parameters"]["termination_gap_corr"] = 5

    if force_sparse or super_fast:
        vmat_opt_params.setdefault("opt_parameters", {})["flag_full_matrix"] = 0

    if opt_params_overrides:
        vmat_opt_params = _apply_opt_params_overrides(vmat_opt_params, opt_params_overrides)
        _log_event(out_dir, "opt_params", "Applied optimization param overrides", data=opt_params_overrides)

    if force_sparse or super_fast:
        vmat_opt_params.setdefault("opt_parameters", {})["flag_full_matrix"] = 0
    flag_full_matrix = vmat_opt_params.get("opt_parameters", {}).get("flag_full_matrix", False)
    run_config.update(
        {
            "use_planner_beams": use_planner_beams,
            "use_available_beams": use_available_beams,
            "force_sparse": force_sparse,
            "super_fast": super_fast,
            "flag_full_matrix": bool(flag_full_matrix),
            "opt_params_file": str(opt_params_path),
            "clinical_criteria_file": str(criteria_path),
        }
    )
    _write_json(out_dir / "config.json", run_config)

    structs = pp.Structures(data)
    _stage_begin("beams", "Selecting beams")
    beam_ids = None
    if use_planner_beams:
        beam_ids = _load_planner_beam_ids(patient_dir)
        if beam_ids:
            print(f"Using PlannerBeams.json beam IDs ({len(beam_ids)} beams).")
            _log_event(out_dir, "beams", f"Using PlannerBeams.json ({len(beam_ids)} beams)")
    if beam_ids is None and use_available_beams:
        beam_ids = _available_beam_ids(patient_dir)
        if beam_ids:
            print(f"Using available beam IDs from data ({len(beam_ids)} beams).")
            _log_event(out_dir, "beams", f"Using available beams ({len(beam_ids)} beams)")
    if beam_ids is None:
        beam_ids = list(range(37))
        _log_event(out_dir, "beams", "Using default beam IDs (0-36)")
    all_beam_ids = np.array(beam_ids)
    arcs_dict = {
        "arcs": [
            {"arc_id": "01", "beam_ids": all_beam_ids[0:int(len(all_beam_ids) / 2)]},
            {"arc_id": "02", "beam_ids": all_beam_ids[int(len(all_beam_ids) / 2):]},
        ]
    }
    beam_ids = [beam_id for arc in arcs_dict["arcs"] for beam_id in arc["beam_ids"]]
    beams = pp.Beams(data, beam_ids=beam_ids, load_inf_matrix_full=flag_full_matrix)
    timing["beams_sec"] = _stage_end(
        "beams",
        f"Beam setup complete ({len(beam_ids)} beams)",
        extra={"beam_count": len(beam_ids)},
    )
    timing["beams_rss_mb"] = _rss_mb()

    if "Patient Surface" in structs.get_structures():
        ind = structs.structures_dict["name"].index("Patient Surface")
        structs.structures_dict["name"][ind] = "BODY"

    for i in range(len(structs.structures_dict["name"])):
        structs.structures_dict["name"][i] = structs.structures_dict["name"][i].upper()

    for i in range(len(vmat_opt_params["steps"])):
        structs.create_opt_structures(
            opt_params=vmat_opt_params["steps"][str(i + 1)],
            clinical_criteria=clinical_criteria,
        )
    timing["case_load_sec"] = _stage_end(
        "case_load",
        f"Loaded case data in {time.perf_counter() - t_case_start:.2f}s",
    )
    timing["case_load_rss_mb"] = _rss_mb()

    _stage_begin("ddc", "Building influence matrix (this can take several minutes)")
    t_ddc_start = time.perf_counter()
    heartbeat = _start_heartbeat(out_dir, "ddc", "Influence matrix build in progress")
    try:
        inf_matrix = pp.InfluenceMatrix(structs=structs, beams=beams, is_full=flag_full_matrix)
    finally:
        heartbeat.set()
    _ensure_voxel_coordinates(inf_matrix)

    if flag_full_matrix:
        A = deepcopy(inf_matrix.A)
        threshold_perc = vmat_opt_params["opt_parameters"].get("threshold_perc", 5)
        sparsification = vmat_opt_params["opt_parameters"].get("sparsification", "Naive")
        B = get_sparse_only(A, threshold_perc=threshold_perc, compression=sparsification)
        inf_matrix.A = B
    timing["ddc_load_sec"] = time.perf_counter() - t_ddc_start
    timing["ddc_rss_mb"] = _rss_mb()
    _stage_end("ddc", f"Influence matrix ready in {timing['ddc_load_sec']:.2f}s")

    inf_matrix_scale_factor = vmat_opt_params["opt_parameters"].get("inf_matrix_scale_factor", 1)
    print("inf_matrix_scale_factor: ", inf_matrix_scale_factor)
    inf_matrix.A = inf_matrix.A * np.float32(inf_matrix_scale_factor)

    arcs = Arcs(arcs_dict=arcs_dict, inf_matrix=inf_matrix)
    my_plan = pp.Plan(
        structs=structs,
        beams=beams,
        inf_matrix=inf_matrix,
        clinical_criteria=clinical_criteria,
        arcs=arcs,
    )

    _stage_begin("optimization", "Running optimization")
    t_opt_start = time.perf_counter()
    opt_detail: dict[str, float] = {}
    if vmat_opt_params["opt_parameters"]["initial_leaf_pos"].lower() == "cg":
        start_col_gen = time.time()
        vmat_opt = EchoVmatOptimizationColGen(my_plan=my_plan, opt_params=vmat_opt_params, step_num=1)
        sol_col_gen = vmat_opt.run_col_gen_algo(solver=solver, verbose=True, accept_unknown=True)
        dose_1d = inf_matrix.A @ sol_col_gen["optimal_intensity"] * my_plan.get_num_of_fractions()
        fig, ax = plt.subplots(figsize=(12, 8))
        struct_names = [
            "PTV",
            "ESOPHAGUS",
            "HEART",
            "CORD",
            "RIND_0",
            "RIND_1",
            "LUNGS_NOT_GTV",
            "RECT_WALL",
            "BLAD_WALL",
            "URETHRA",
        ]
        ax = pp.Visualization.plot_dvh(my_plan, dose_1d=dose_1d, struct_names=struct_names, ax=ax)
        ax.set_title("Initial Col gen dvh")
        fig.savefig(out_dir / "dvh_initial_col_gen.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        end_col_gen = time.time()
        opt_detail["col_gen_sec"] = end_col_gen - start_col_gen
        print(
            "***************time to generate initial leaf positions = ",
            end_col_gen - start_col_gen,
            "seconds *****************",
        )

    clinical_criteria.get_dvh_table(my_plan=my_plan, opt_params=vmat_opt_params["steps"]["2"])
    vmat_opt = EchoVmatOptimization(my_plan=my_plan, opt_params=vmat_opt_params, step_num=1)

    solutions = []
    sol = {}
    final_convergence = []
    if not clinical_criteria.dvh_table.empty:
        sol_convergence = vmat_opt.run_sequential_cvx_algo(solver=solver, verbose=True)
        _emit_trace(out_dir, sol_convergence, "step_0")
        final_convergence.extend(sol_convergence)
        sol = sol_convergence[vmat_opt.best_iteration]
        solutions.append(sol)
        vmat_opt.update_params(step_number=0, sol=sol)

    for i in range(2):
        step_time = time.time()
        if not clinical_criteria.dvh_table.empty:
            dose = sol["act_dose_v"] * my_plan.get_num_of_fractions()
            clinical_criteria.get_low_dose_vox_ind(my_plan, dose=dose, inf_matrix=my_plan.inf_matrix)
        vmat_opt.set_step_num(i + 1)
        sol_convergence = vmat_opt.run_sequential_cvx_algo(solver=solver, verbose=True)
        _emit_trace(out_dir, sol_convergence, f"step_{i + 1}")
        final_convergence.extend(sol_convergence)
        sol = final_convergence[vmat_opt.best_iteration]
        solutions.append(sol)
        vmat_opt.update_params(step_number=i + 1, sol=sol)
        print("***************Time for step {}:{} *******************".format(i + 1, time.time() - step_time))

    timing["optimization_sec"] = time.perf_counter() - t_opt_start
    timing["optimization_detail"] = opt_detail
    timing["optimization_rss_mb"] = _rss_mb()
    _stage_end("optimization", f"Optimization finished in {timing['optimization_sec']:.2f}s")

    fig, ax = plt.subplots(figsize=(12, 8))
    struct_names = [
        "PTV",
        "ESOPHAGUS",
        "HEART",
        "CORD",
        "RIND_0",
        "RIND_1",
        "LUNGS_NOT_GTV",
        "RECT_WALL",
        "BLAD_WALL",
        "URETHRA",
        "LUNG_L",
        "LUNG_R",
    ]
    title = []
    style = ["-", "--", ":"]
    for i in range(len(solutions)):
        ax = pp.Visualization.plot_dvh(
            my_plan,
            dose_1d=solutions[i]["act_dose_v"] * my_plan.get_num_of_fractions(),
            struct_names=struct_names,
            style=style[i],
            ax=ax,
        )
        if len(solutions) < 3:
            title.append(f"Step {i + 1} {style[i]}")
        else:
            title.append(f"Step {i} {style[i]}")
    ax.set_title(" ".join(title))
    fig.savefig(out_dir / "dvh_steps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    solver_trace = []
    for sol in final_convergence:
        solver_trace.append(
            {
                "outer_iteration": sol.get("outer_iteration"),
                "inner_iteration": sol.get("inner_iteration"),
                "intermediate_obj_value": sol.get("intermediate_obj_value"),
                "actual_obj_value": sol.get("actual_obj_value"),
                "accept": sol.get("accept"),
                "forward_backward": sol.get("forward_backward"),
                "step_size_f_b": sol.get("step_size_f_b"),
            }
        )
    _write_json(out_dir / "solver_trace.json", solver_trace)

    timing["correction_sec"] = 0.0

    _stage_begin("evaluation", "Evaluating plan metrics")
    t_eval_start = time.perf_counter()
    dose_1d = solutions[-1]["act_dose_v"] * my_plan.get_num_of_fractions()
    metrics_df = Evaluation.display_clinical_criteria(
        my_plan=my_plan,
        sol=solutions[-1],
        dose_1d=dose_1d,
        clinical_criteria=clinical_criteria,
        return_df=True,
        in_browser=False,
        open_browser=False,
    )
    metrics_records = []
    metrics_columns = []
    if metrics_df is not None:
        if hasattr(metrics_df, "data"):
            metrics_df = metrics_df.data
        elif hasattr(metrics_df, "_data"):
            metrics_df = metrics_df._data
        if hasattr(metrics_df, "to_dict"):
            metrics_columns = list(metrics_df.columns)
            metrics_records = metrics_df.to_dict(orient="records")
        else:
            _log_event(
                out_dir,
                "evaluation",
                f"Skipping metrics export; unexpected type: {type(metrics_df)}",
                level="warning",
            )
    _write_json(
        out_dir / "metrics.json",
        {
            "metrics": metrics_records,
            "columns": metrics_columns,
            "records": metrics_records,
        },
    )
    try:
        Evaluation.display_clinical_criteria(
            my_plan=my_plan,
            sol=solutions[-1],
            dose_1d=dose_1d,
            clinical_criteria=clinical_criteria,
            html_file_name="clinical_criteria.html",
            return_df=False,
            in_browser=True,
            path=str(out_dir),
            open_browser=False,
        )
    except Exception as exc:
        _log_event(
            out_dir,
            "evaluation",
            f"Failed to write clinical_criteria.html: {exc}",
            level="warning",
        )
    _write_json(out_dir / "clinical_criteria.json", clinical_criteria.clinical_criteria_dict)

    dvh = {}
    dvh_sol = {"inf_matrix": my_plan.inf_matrix, "dose_1d": dose_1d}
    for struct_name in _select_struct_names(my_plan):
        x, y = Evaluation.get_dvh(sol=dvh_sol, struct=struct_name, dose_1d=dose_1d)
        dvh[struct_name] = {
            "dose_gy": x.tolist(),
            "volume_fraction": y.tolist(),
        }
    _write_json(out_dir / "dvh.json", dvh)
    timing["evaluation_sec"] = time.perf_counter() - t_eval_start
    timing["evaluation_rss_mb"] = _rss_mb()
    _stage_end("evaluation", f"Evaluation finished in {timing['evaluation_sec']:.2f}s")
    try:
        try:
            from backend.plan_score.score import compute_plan_score_for_run
        except ImportError:
            from plan_score.score import compute_plan_score_for_run
        plan_score = compute_plan_score_for_run(out_dir, protocol_name)
        _write_json(out_dir / "plan_score.json", plan_score)
    except Exception as exc:
        _log_event(
            out_dir,
            "plan_score",
            f"Plan score not generated: {exc}",
            level="warning",
        )

    print("saving optimal solution..")
    for i in range(len(solutions)):
        sol_name = f"sol_step{i + 1}.pkl" if len(solutions) < 3 else f"sol_step{i}.pkl"
        pp.save_optimal_sol(sol=solutions[i], sol_name=sol_name, path=str(out_dir))

    print("saving my_plan..")
    pp.save_plan(my_plan, "my_plan.pkl", path=str(out_dir))

    opt_time = round(time.time() - tic_all, 2)
    print("***************** opt_time (secs) ********************:", opt_time)
    timing["total_sec"] = time.perf_counter() - t_total_start
    timing["opt_time_sec"] = opt_time
    if rss_samples:
        timing["max_rss_mb"] = max(rss_samples)
    _write_json(out_dir / "timing.json", timing)
    elapsed_total = time.perf_counter() - run_start
    rss_mb = _rss_mb()
    _update_status(
        out_dir,
        {
            "state": "completed",
            "stage": "complete",
            "started_at": started_at,
            "ended_at": _now_iso(),
            "elapsed_sec": elapsed_total,
            "rss_mb": rss_mb,
            "case_id": case_id,
            "protocol": protocol_name,
        },
    )
    _log_event(out_dir, "complete", "Run completed")
    return out_dir


def run_compressrtp(
    case_id: str = "Lung_Phantom_Patient_1",
    protocol_name: str = "Lung_2Gy_30Fx",
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    solver: str = "MOSEK",
    use_planner_beams: bool = False,
    use_available_beams: bool = False,
    beam_ids_override: list[int] | None = None,
    compress_mode: str = "sparse-only",
    threshold_perc: float = 10.0,
    rank: int = 5,
    step: str | None = None,
    fast: bool = False,
    super_fast: bool = False,
    use_gpu: bool = False,
    opt_params_overrides: dict | None = None,
    tag: str | None = None,
    adapter=None,
) -> Path:
    _ensure_compressrtp_on_path()
    import cvxpy as cp
    from compress_rtp.compress_rtp_optimization import CompressRTPOptimization
    from compress_rtp.utils.get_low_dim_basis import get_low_dim_basis
    from compress_rtp.utils.get_sparse_only import get_sparse_only
    from compress_rtp.utils.get_sparse_plus_low_rank import get_sparse_plus_low_rank
    from portpy.photon.evaluation import Evaluation
    try:
        from backend.adapters.example_adapter import ExampleAdapter
    except ImportError:
        from adapters.example_adapter import ExampleAdapter

    data_dir = _normalize_data_dir(data_dir or _default_data_dir(), case_id)
    out_dir = out_dir or _default_out_dir(case_id, run_type="compressrtp")
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_dir = data_dir / case_id

    if adapter is None:
        adapter = ExampleAdapter(data_root=data_dir)
    case_info = adapter.prepare_case(case_id)
    data_dir = _normalize_data_dir(case_info.data_dir, case_id)
    patient_dir = data_dir / case_id

    if fast and not super_fast:
        threshold_perc = max(threshold_perc, 12.0)
        rank = min(rank, 3)
    if super_fast:
        use_planner_beams = True
        use_available_beams = True
        threshold_perc = max(threshold_perc, 15.0)
        rank = min(rank, 2)

    step = None if step in (None, "all") else step
    run_config = {
        "case_id": case_id,
        "protocol": protocol_name,
        "solver": solver,
        "data_dir": str(data_dir),
        "optimizer": "compressrtp",
        "compress_mode": compress_mode,
        "threshold_perc": float(threshold_perc),
        "rank": int(rank),
        "step": step,
        "fast": fast,
        "super_fast": super_fast,
        "use_gpu": use_gpu,
        "adapter": case_info.source,
        "dataset_download_sec": float(case_info.download_seconds),
    }
    if tag:
        run_config["tag"] = tag
    if opt_params_overrides:
        run_config["opt_params_overrides"] = opt_params_overrides
    if beam_ids_override:
        run_config["beam_ids_override"] = beam_ids_override
    _write_json(out_dir / "config.json", run_config)
    started_at = _now_iso()
    _update_status(
        out_dir,
        {
            "state": "running",
            "started_at": started_at,
            "case_id": case_id,
            "protocol": protocol_name,
            "optimizer": "compressrtp",
        },
    )
    _log_event(out_dir, "start", "CompressRTP run started")

    run_start = time.perf_counter()
    stage_starts: dict[str, float] = {}
    rss_samples: list[float] = []

    def _stage_begin(stage: str, message: str | None = None) -> None:
        stage_starts[stage] = time.perf_counter()
        elapsed = stage_starts[stage] - run_start
        rss_mb = _rss_mb()
        if rss_mb is not None:
            rss_samples.append(rss_mb)
        _merge_status(out_dir, {"stage": stage, "elapsed_sec": elapsed, "rss_mb": rss_mb})
        _log_event(
            out_dir,
            stage,
            message or f"{stage} started",
            data={"state": "start", "elapsed_sec": elapsed, "rss_mb": rss_mb},
        )

    def _stage_end(stage: str, message: str, extra: dict | None = None) -> float:
        end = time.perf_counter()
        start = stage_starts.get(stage, end)
        elapsed = end - run_start
        duration = end - start
        rss_mb = _rss_mb()
        if rss_mb is not None:
            rss_samples.append(rss_mb)
        data = {"state": "done", "elapsed_sec": elapsed, "duration_sec": duration, "rss_mb": rss_mb}
        if extra:
            data.update(extra)
        _log_event(out_dir, stage, message, data=data)
        return duration

    tic_all = time.time()
    t_total_start = time.perf_counter()
    timing: dict[str, float | dict] = {
        "dataset_download_sec": float(case_info.download_seconds),
    }

    def _finish_step(step_name: str, info: dict | None = None) -> Path:
        timing["total_sec"] = time.perf_counter() - t_total_start
        timing["step"] = step_name
        if info:
            timing["step_info"] = info
        if rss_samples:
            timing["max_rss_mb"] = max(rss_samples)
        _write_json(out_dir / "timing.json", timing)
        elapsed_total = time.perf_counter() - run_start
        rss_mb = _rss_mb()
        _merge_status(
            out_dir,
            {
                "state": "completed",
                "stage": step_name,
                "started_at": started_at,
                "ended_at": _now_iso(),
                "elapsed_sec": elapsed_total,
                "rss_mb": rss_mb,
                "case_id": case_id,
                "protocol": protocol_name,
                "optimizer": "compressrtp",
                "step": step_name,
            },
        )
        _log_event(out_dir, "complete", f"Run completed (stop after {step_name})")
        return out_dir
    if case_info.download_seconds:
        _log_event(out_dir, "dataset_download", f"Downloaded dataset in {case_info.download_seconds:.2f}s")

    _stage_begin("case_load", "Loading case data")
    t_case_start = time.perf_counter()
    data = pp.DataExplorer(data_dir=str(data_dir))
    data.patient_id = case_id
    ct = pp.CT(data)
    structs = pp.Structures(data)
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)
    opt_params = data.load_config_opt_params(protocol_name=protocol_name)
    if opt_params_overrides:
        opt_params = _apply_opt_params_overrides(opt_params, opt_params_overrides)
        _log_event(out_dir, "opt_params", "Applied optimization param overrides", data=opt_params_overrides)
    structs.create_opt_structures(opt_params=opt_params, clinical_criteria=clinical_criteria)
    timing["case_load_sec"] = time.perf_counter() - t_case_start
    timing["case_load_rss_mb"] = _rss_mb()
    _stage_end("case_load", f"Loaded case data in {timing['case_load_sec']:.2f}s")

    _stage_begin("beams", "Selecting beams")
    beam_ids = None
    if beam_ids_override is not None:
        beam_ids = beam_ids_override
        _log_event(out_dir, "beams", f"Using override beam IDs ({len(beam_ids)} beams)")
    elif use_planner_beams:
        beam_ids = _load_planner_beam_ids(patient_dir)
        if beam_ids:
            _log_event(out_dir, "beams", f"Using PlannerBeams.json ({len(beam_ids)} beams)")
    if beam_ids is None and use_available_beams:
        beam_ids = _available_beam_ids(patient_dir)
        if beam_ids:
            _log_event(out_dir, "beams", f"Using available beams ({len(beam_ids)} beams)")
    if beam_ids:
        beams = pp.Beams(data, beam_ids=beam_ids, load_inf_matrix_full=True)
    else:
        beams = pp.Beams(data, load_inf_matrix_full=True)
        _log_event(out_dir, "beams", "Using planner beams (default)")
    timing["beams_sec"] = _stage_end(
        "beams",
        f"Beam setup complete ({len(beam_ids) if beam_ids else beams.get_num_beams()} beams)",
        extra={"beam_count": len(beam_ids) if beam_ids else beams.get_num_beams()},
    )
    timing["beams_rss_mb"] = _rss_mb()

    _stage_begin("ddc", "Building influence matrix (this can take several minutes)")
    t_ddc_start = time.perf_counter()
    heartbeat = _start_heartbeat(out_dir, "ddc", "Influence matrix build in progress", 60.0)
    try:
        inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams, is_full=True)
    finally:
        heartbeat.set()
    timing["ddc_load_sec"] = time.perf_counter() - t_ddc_start
    timing["ddc_rss_mb"] = _rss_mb()
    _stage_end("ddc", f"Influence matrix ready in {timing['ddc_load_sec']:.2f}s")
    if step == "ddc":
        info = _matrix_info(inf_matrix.A)
        _log_event(out_dir, "ddc", "DDC matrix ready", data=info)
        return _finish_step("ddc", info)

    my_plan = pp.Plan(
        ct=ct,
        structs=structs,
        beams=beams,
        inf_matrix=inf_matrix,
        clinical_criteria=clinical_criteria,
    )

    num_fractions = clinical_criteria.get_num_of_fractions()
    A_full = inf_matrix.A
    gpu_ctx = None
    gpu_modules = {}
    gpu_chunk_mb = None
    gpu_compress_chunk_mb = None
    gpu_compress_chunk_rows = None
    if use_gpu:
        gpu_ctx, gpu_modules = get_gpu_context(use_gpu=True)
        if gpu_ctx.enabled:
            try:
                gpu_chunk_mb = int(os.environ.get("ECHO_GPU_CHUNK_MB", "512"))
            except ValueError:
                gpu_chunk_mb = 512
            try:
                gpu_compress_chunk_mb = int(os.environ.get("ECHO_GPU_COMPRESS_CHUNK_MB", "256"))
            except ValueError:
                gpu_compress_chunk_mb = 256
            gpu_compress_chunk_rows = estimate_chunk_rows(A_full, gpu_compress_chunk_mb)
            _log_event(
                out_dir,
                "gpu",
                f"GPU enabled{f': {gpu_ctx.name}' if gpu_ctx.name else ''}",
                data={
                    "free_mb": gpu_ctx.free_mb,
                    "total_mb": gpu_ctx.total_mb,
                    "chunk_mb": gpu_chunk_mb,
                    "compress_chunk_mb": gpu_compress_chunk_mb,
                    "compress_chunk_rows": gpu_compress_chunk_rows,
                },
            )
        else:
            _log_event(
                out_dir,
                "gpu",
                "GPU unavailable, falling back to CPU",
                level="warning",
                data={"reason": gpu_ctx.reason},
            )
    gpu_enabled = bool(gpu_ctx and gpu_ctx.enabled)
    gpu_xp = gpu_modules.get("xp") if gpu_enabled else None

    def _build_sparse_plus_low_rank(use_gpu_flag: bool):
        try:
            result = get_sparse_plus_low_rank(
                A=A_full,
                threshold_perc=threshold_perc,
                rank=rank,
                xp=gpu_xp if use_gpu_flag else None,
                chunk_rows=gpu_compress_chunk_rows if use_gpu_flag else None,
            )
            if use_gpu_flag:
                data = {"stage": "compress"}
                if gpu_compress_chunk_rows is not None and gpu_compress_chunk_rows < A_full.shape[0]:
                    data.update(
                        {
                            "chunk_rows": int(gpu_compress_chunk_rows),
                            "chunk_mb": gpu_compress_chunk_mb,
                        }
                    )
                _log_event(
                    out_dir,
                    "gpu",
                    "GPU used for sparse-plus-low-rank thresholding",
                    data=data,
                )
            return result
        except Exception as exc:  # pragma: no cover - runtime fallback
            if use_gpu_flag:
                _log_event(
                    out_dir,
                    "gpu",
                    "GPU compression failed, falling back to CPU",
                    level="warning",
                    data={"error": str(exc)},
                )
                return get_sparse_plus_low_rank(
                    A=A_full,
                    threshold_perc=threshold_perc,
                    rank=rank,
                    xp=None,
                )
            raise
    _stage_begin("compress", "Building compressed representation")
    if gpu_enabled and compress_mode != "sparse-plus-low-rank" and step not in ("svd",):
        _log_event(
            out_dir,
            "gpu",
            "GPU compression not supported for this mode; using CPU",
            data={"compress_mode": compress_mode, "step": step},
        )
    if step == "sparse":
        S_sparse = get_sparse_only(A=A_full, threshold_perc=threshold_perc, compression="rmr")
        info = {"sparse": _matrix_info(S_sparse)}
        timing["compress_sec"] = _stage_end("compress", "Sparse-only matrix built", extra=info)
        timing["compress_rss_mb"] = _rss_mb()
        return _finish_step("sparse", info)
    if step == "svd":
        S, H, W = _build_sparse_plus_low_rank(gpu_enabled)
        info = {
            "sparse": _matrix_info(S),
            "H_shape": list(H.shape),
            "W_shape": list(W.shape),
        }
        timing["compress_sec"] = _stage_end("compress", "Sparse + low-rank factors built", extra=info)
        timing["compress_rss_mb"] = _rss_mb()
        return _finish_step("svd", info)
    if step == "wavelet":
        S_sparse = get_sparse_only(A=A_full, threshold_perc=threshold_perc, compression="rmr")
        inf_matrix.A = S_sparse
        basis = get_low_dim_basis(inf_matrix=inf_matrix, compression="wavelet")
        info = {
            "sparse": _matrix_info(S_sparse),
            "basis_shape": list(basis.shape),
        }
        timing["compress_sec"] = _stage_end("compress", "Wavelet basis built", extra=info)
        timing["compress_rss_mb"] = _rss_mb()
        return _finish_step("wavelet", info)

    compress_info: dict | None = None
    opt = None
    solve_kwargs = {"solver": solver, "verbose": True}
    solver_trace = []
    if compress_mode == "sparse-only":
        S_sparse = get_sparse_only(A=A_full, threshold_perc=threshold_perc, compression="rmr")
        compress_info = {"sparse": _matrix_info(S_sparse)}
        inf_matrix.A = S_sparse
        opt = pp.Optimization(my_plan, inf_matrix=inf_matrix, opt_params=opt_params)
        opt.create_cvxpy_problem()
    elif compress_mode == "sparse-plus-low-rank":
        S, H, W = _build_sparse_plus_low_rank(gpu_enabled)
        compress_info = {
            "sparse": _matrix_info(S),
            "H_shape": list(H.shape),
            "W_shape": list(W.shape),
        }
        opt = CompressRTPOptimization(my_plan, opt_params=opt_params)
        opt.create_cvxpy_problem_compressed(S=S, H=H, W=W)
        if solver.upper() == "MOSEK":
            solve_kwargs["mosek_params"] = {
                "MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES": 0,
                "MSK_IPAR_INTPNT_SCALING": "MSK_SCALING_NONE",
            }
    elif compress_mode == "wavelet":
        for obj in opt_params.get("objective_functions", []):
            if obj.get("type") == "smoothness-quadratic":
                obj["weight"] = 0
        S_sparse = get_sparse_only(A=A_full, threshold_perc=threshold_perc, compression="rmr")
        compress_info = {"sparse": _matrix_info(S_sparse)}
        inf_matrix.A = S_sparse
        opt = pp.Optimization(my_plan, inf_matrix=inf_matrix, opt_params=opt_params)
        opt.create_cvxpy_problem()
        wavelet_basis = get_low_dim_basis(inf_matrix=inf_matrix, compression="wavelet")
        if compress_info is not None:
            compress_info["basis_shape"] = list(wavelet_basis.shape)
        y = cp.Variable(wavelet_basis.shape[1])
        opt.constraints += [wavelet_basis @ y == opt.vars["x"]]
    else:
        raise ValueError(f"Unknown compress_mode: {compress_mode}")

    timing["compress_sec"] = _stage_end(
        "compress",
        "Compressed representation ready",
        extra=compress_info,
    )
    timing["compress_rss_mb"] = _rss_mb()

    _stage_begin("optimization", "Running optimization")
    t_opt_start = time.perf_counter()
    sol = opt.solve(**solve_kwargs)
    t_dose_start = time.perf_counter()
    if gpu_enabled:
        chunk_rows = estimate_chunk_rows(A_full, gpu_chunk_mb)
        if chunk_rows is not None and chunk_rows < A_full.shape[0]:
            _log_event(
                out_dir,
                "gpu",
                "Using chunked GPU matmul for dose",
                data={"chunk_rows": int(chunk_rows), "chunk_mb": gpu_chunk_mb},
            )
        try:
            dose_1d = gpu_matmul(
                A_full,
                sol["optimal_intensity"],
                xp=gpu_modules["xp"],
                sparse_module=gpu_modules["sparse"],
                max_chunk_mb=gpu_chunk_mb,
            )
            dose_1d = dose_1d * num_fractions
            _log_event(out_dir, "gpu", "Dose computed on GPU", data={"stage": "dose_1d"})
        except Exception as exc:  # pragma: no cover - depends on GPU state
            _log_event(
                out_dir,
                "gpu",
                "GPU dose computation failed; using CPU",
                level="warning",
                data={"error": str(exc)},
            )
            dose_1d = (A_full @ sol["optimal_intensity"]) * num_fractions
    else:
        dose_1d = (A_full @ sol["optimal_intensity"]) * num_fractions
    timing["dose_1d_sec"] = time.perf_counter() - t_dose_start
    timing["optimization_sec"] = time.perf_counter() - t_opt_start
    timing["optimization_rss_mb"] = _rss_mb()
    _stage_end("optimization", f"Optimization finished in {timing['optimization_sec']:.2f}s")
    _write_json(out_dir / "solver_trace.json", solver_trace)

    t_eval_start = time.perf_counter()
    metrics_df = Evaluation.display_clinical_criteria(
        my_plan=my_plan,
        sol=sol,
        dose_1d=dose_1d,
        clinical_criteria=clinical_criteria,
        return_df=True,
        in_browser=False,
        open_browser=False,
    )
    metrics_records = []
    metrics_columns = []
    if metrics_df is not None:
        if hasattr(metrics_df, "data"):
            metrics_df = metrics_df.data
        elif hasattr(metrics_df, "_data"):
            metrics_df = metrics_df._data
        if hasattr(metrics_df, "to_dict"):
            metrics_columns = list(metrics_df.columns)
            metrics_records = metrics_df.to_dict(orient="records")
        else:
            _log_event(
                out_dir,
                "evaluation",
                f"Skipping metrics export; unexpected type: {type(metrics_df)}",
                level="warning",
            )
    _write_json(
        out_dir / "metrics.json",
        {
            "metrics": metrics_records,
            "columns": metrics_columns,
            "records": metrics_records,
        },
    )
    try:
        Evaluation.display_clinical_criteria(
            my_plan=my_plan,
            sol=sol,
            dose_1d=dose_1d,
            clinical_criteria=clinical_criteria,
            html_file_name="clinical_criteria.html",
            return_df=False,
            in_browser=True,
            path=str(out_dir),
            open_browser=False,
        )
    except Exception as exc:
        _log_event(
            out_dir,
            "evaluation",
            f"Failed to write clinical_criteria.html: {exc}",
            level="warning",
        )
    _write_json(out_dir / "clinical_criteria.json", clinical_criteria.clinical_criteria_dict)

    fig, ax = plt.subplots(figsize=(12, 8))
    struct_names = _select_struct_names(my_plan)
    ax = pp.Visualization.plot_dvh(my_plan, dose_1d=dose_1d, struct_names=struct_names, ax=ax)
    ax.set_title(f"CompressRTP DVH ({compress_mode})")
    fig.savefig(out_dir / "dvh_steps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    dvh = {}
    dvh_sol = {"inf_matrix": my_plan.inf_matrix, "dose_1d": dose_1d}
    for struct_name in _select_struct_names(my_plan):
        x, y = Evaluation.get_dvh(sol=dvh_sol, struct=struct_name, dose_1d=dose_1d)
        dvh[struct_name] = {
            "dose_gy": x.tolist(),
            "volume_fraction": y.tolist(),
        }
    _write_json(out_dir / "dvh.json", dvh)
    timing["evaluation_sec"] = time.perf_counter() - t_eval_start
    timing["evaluation_rss_mb"] = _rss_mb()
    _stage_end("evaluation", f"Evaluation finished in {timing['evaluation_sec']:.2f}s")
    try:
        try:
            from backend.plan_score.score import compute_plan_score_for_run
        except ImportError:
            from plan_score.score import compute_plan_score_for_run
        plan_score = compute_plan_score_for_run(out_dir, protocol_name)
        _write_json(out_dir / "plan_score.json", plan_score)
    except Exception as exc:
        _log_event(
            out_dir,
            "plan_score",
            f"Plan score not generated: {exc}",
            level="warning",
        )

    pp.save_optimal_sol(sol=sol, sol_name="sol_step1.pkl", path=str(out_dir))
    pp.save_plan(my_plan, "my_plan.pkl", path=str(out_dir))

    opt_time = round(time.time() - tic_all, 2)
    timing["total_sec"] = time.perf_counter() - t_total_start
    timing["opt_time_sec"] = opt_time
    if rss_samples:
        timing["max_rss_mb"] = max(rss_samples)
    _write_json(out_dir / "timing.json", timing)
    elapsed_total = time.perf_counter() - run_start
    rss_mb = _rss_mb()
    _update_status(
        out_dir,
        {
            "state": "completed",
            "stage": "complete",
            "started_at": started_at,
            "ended_at": _now_iso(),
            "elapsed_sec": elapsed_total,
            "rss_mb": rss_mb,
            "case_id": case_id,
            "protocol": protocol_name,
            "optimizer": "compressrtp",
        },
    )
    _log_event(out_dir, "complete", "CompressRTP run completed")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ECHO-VMAT or CompressRTP examples with timing + artifacts.")
    parser.add_argument("--case-id", default="Lung_Phantom_Patient_1")
    parser.add_argument("--protocol", default="Lung_2Gy_30Fx")
    parser.add_argument("--data-dir", default=str(_default_data_dir()))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument(
        "--optimizer",
        default="echo-vmat",
        choices=["echo-vmat", "compressrtp"],
        help="Select optimization pipeline.",
    )
    parser.add_argument(
        "--compress-mode",
        default="sparse-only",
        choices=["sparse-only", "sparse-plus-low-rank", "wavelet"],
        help="CompressRTP compression mode.",
    )
    parser.add_argument("--threshold-perc", type=float, default=10.0)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "ddc", "sparse", "svd", "wavelet"],
        help="Stop after a CompressRTP pipeline step (for diagnostics).",
    )
    parser.add_argument("--tag", default=None, help="Optional run tag for labeling.")
    parser.add_argument(
        "--opt-params-overrides",
        default=None,
        help="JSON string or @path for optimization parameter overrides.",
    )
    parser.add_argument(
        "--beam-ids",
        default=None,
        help="Comma-separated beam IDs override (e.g. 0,1,2).",
    )
    parser.add_argument("--adapter", default="example", choices=["example", "huggingface"])
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-subdir", default=None)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Reduce memory/time (planner beams + sparse influence matrix).",
    )
    parser.add_argument(
        "--super-fast",
        action="store_true",
        help="Very fast run (planner beams + skip CG + fewer SCP iterations + sparse influence matrix).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration for supported CompressRTP steps.",
    )
    parser.add_argument(
        "--use-planner-beams",
        action="store_true",
        help="Use PlannerBeams.json beam IDs when available.",
    )
    parser.add_argument(
        "--use-available-beams",
        action="store_true",
        help="Use beam IDs discovered from Beams/Beam_*_MetaData.json.",
    )
    parser.add_argument(
        "--force-sparse",
        action="store_true",
        help="Force sparse influence matrix loading (overrides flag_full_matrix).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None
    use_planner_beams = args.use_planner_beams or args.fast or args.super_fast
    use_available_beams = args.use_available_beams or args.fast or args.super_fast
    force_sparse = args.force_sparse or args.fast or args.super_fast
    super_fast = args.super_fast
    beam_ids_override = _parse_beam_ids(args.beam_ids)
    adapter = None
    if args.adapter == "huggingface":
        try:
            from backend.adapters.huggingface_adapter import HuggingFaceAdapter
        except ImportError:
            from adapters.huggingface_adapter import HuggingFaceAdapter
        if not args.hf_repo_id:
            raise SystemExit("--hf-repo-id is required when --adapter huggingface is used.")
        adapter = HuggingFaceAdapter(
            repo_id=args.hf_repo_id,
            cache_dir=_repo_root() / "data" / "raw" / "huggingface",
            token=args.hf_token,
            subdir=args.hf_subdir,
        )
    opt_params_overrides = _parse_opt_params_overrides(args.opt_params_overrides)
    run_echo_example(
        case_id=args.case_id,
        protocol_name=args.protocol,
        data_dir=data_dir,
        out_dir=out_dir,
        solver=args.solver,
        use_planner_beams=use_planner_beams,
        use_available_beams=use_available_beams,
        force_sparse=force_sparse,
        super_fast=super_fast,
        opt_params_overrides=opt_params_overrides,
        tag=args.tag,
        adapter=adapter,
    ) if args.optimizer == "echo-vmat" else run_compressrtp(
        case_id=args.case_id,
        protocol_name=args.protocol,
        data_dir=data_dir,
        out_dir=out_dir,
        solver=args.solver,
        use_planner_beams=use_planner_beams,
        use_available_beams=use_available_beams,
        beam_ids_override=beam_ids_override,
        compress_mode=args.compress_mode,
        threshold_perc=args.threshold_perc,
        rank=args.rank,
        step=args.step,
        fast=args.fast,
        super_fast=super_fast,
        use_gpu=args.gpu,
        opt_params_overrides=opt_params_overrides,
        tag=args.tag,
        adapter=adapter,
    )


if __name__ == "__main__":
    main()
