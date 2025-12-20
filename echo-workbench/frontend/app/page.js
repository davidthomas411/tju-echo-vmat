"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const CASE_OPTIONS = [
  { id: "Lung_Patient_11", label: "Lung Patient 11" },
  { id: "Lung_Patient_4", label: "Lung Patient 4" },
  { id: "Lung_Patient_2", label: "Lung Patient 2" },
  { id: "Lung_Phantom_Patient_1", label: "Lung Phantom Patient 1" },
];

const PRESETS = [
  { id: "super_fast", label: "Super Fast" },
  { id: "fast", label: "Fast" },
  { id: "balanced", label: "Balanced" },
];

const STAGES = [
  { id: "dataset_download", label: "Dataset" },
  { id: "case_load", label: "Case Load" },
  { id: "beams", label: "Beam Setup" },
  { id: "ddc", label: "DDC" },
  { id: "optimization", label: "Optimization" },
  { id: "evaluation", label: "Evaluation" },
  { id: "complete", label: "Complete" },
];

const TIMING_LABELS = [
  { key: "dataset_download_sec", label: "Dataset" },
  { key: "case_load_sec", label: "Case Load" },
  { key: "ddc_load_sec", label: "DDC" },
  { key: "optimization_sec", label: "Optimization" },
  { key: "evaluation_sec", label: "Evaluation" },
  { key: "total_sec", label: "Total" },
];

const DVH_COLORS = [
  "#60a5fa",
  "#f97316",
  "#22d3ee",
  "#facc15",
  "#34d399",
  "#f472b6",
  "#a78bfa",
  "#f87171",
  "#38bdf8",
  "#fb7185",
  "#4ade80",
  "#e879f9",
];

function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return `${value.toFixed(2)} s`;
}

function extractPlanValue(raw) {
  if (raw === null || raw === undefined) {
    return "";
  }
  if (typeof raw === "number") {
    return raw.toFixed(2);
  }
  if (typeof raw === "string") {
    const match = raw.match(/font-size:13px;">([0-9.+-]+)/);
    if (match) {
      return match[1];
    }
    const fallback = raw.match(/([0-9]+(?:\.[0-9]+)?)/);
    if (fallback) {
      return fallback[1];
    }
  }
  return "";
}

function parseNumeric(value) {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const match = String(value).match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return null;
  }
  return Number.parseFloat(match[0]);
}

function computeStatus(row, planValueDisplay) {
  const planVal = parseNumeric(planValueDisplay ?? row["Plan Value"]);
  const limitVal = parseNumeric(row.Limit);
  const goalVal = parseNumeric(row.Goal);
  if (planVal === null) {
    return "neutral";
  }
  if (limitVal !== null && goalVal !== null) {
    if (limitVal < goalVal) {
      if (planVal >= goalVal) return "good";
      if (planVal >= limitVal) return "warn";
      return "bad";
    }
    if (planVal <= goalVal) return "good";
    if (planVal <= limitVal) return "warn";
    return "bad";
  }
  if (limitVal !== null) {
    return planVal <= limitVal ? "good" : "bad";
  }
  if (goalVal !== null) {
    return planVal <= goalVal ? "good" : "warn";
  }
  return "neutral";
}

function closestIndex(values, target) {
  if (!values.length) {
    return 0;
  }
  let low = 0;
  let high = values.length - 1;
  while (high - low > 1) {
    const mid = Math.floor((low + high) / 2);
    if (values[mid] < target) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return Math.abs(values[low] - target) < Math.abs(values[high] - target) ? low : high;
}

function safeArrayMax(values) {
  let max = 0;
  for (const value of values) {
    if (Number.isFinite(value) && value > max) {
      max = value;
    }
  }
  return max;
}

function arraysEqual(a, b) {
  if (a === b) {
    return true;
  }
  if (!a || !b || a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

function TracePlot({ points }) {
  if (!points.length) {
    return <div className="placeholder">No optimization trace yet.</div>;
  }
  const filtered = points
    .map((point, idx) => ({
      x: point.outer_iteration ?? idx,
      y: point.actual_obj_value ?? point.intermediate_obj_value,
    }))
    .filter((point) => typeof point.y === "number");
  if (!filtered.length) {
    return <div className="placeholder">Trace data is not numeric yet.</div>;
  }

  const width = 520;
  const height = 220;
  const pad = { left: 48, right: 16, top: 18, bottom: 32 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const xs = filtered.map((p) => p.x);
  const ys = filtered.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const path = filtered
    .map((point, idx) => {
      const x = pad.left + ((point.x - minX) / spanX) * plotW;
      const y = pad.top + (1 - (point.y - minY) / spanY) * plotH;
      return `${idx === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="220">
      <rect
        x={pad.left}
        y={pad.top}
        width={plotW}
        height={plotH}
        fill="#0b1220"
        stroke="rgba(148,163,184,0.25)"
        rx="10"
      />
      <path d={path} fill="none" stroke="#5bbcff" strokeWidth="2" />
      <text x={pad.left} y={height - 8} fontSize="10" fill="#94a3b8">
        Iteration
      </text>
      <text x={8} y={pad.top + 12} fontSize="10" fill="#94a3b8">
        Obj
      </text>
    </svg>
  );
}

function DvhPlot({ series, onHover }) {
  if (!series.length) {
    return <div className="placeholder">No DVH available yet.</div>;
  }
  const width = 560;
  const height = 320;
  const pad = { left: 54, right: 20, top: 20, bottom: 36 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxDose = Math.max(...series.map((entry) => entry.maxDose || 0));
  const minDose = 0;
  const volumeScale = series.some((entry) => entry.volumeMax > 1.5) ? 1 : 100;
  const minVol = 0;
  const maxVol = 100;
  const toPercent = (value) => (volumeScale === 1 ? value : value * 100);

  const [hoverX, setHoverX] = useState(null);

  function handleMove(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const clamped = Math.min(Math.max(x, pad.left), pad.left + plotW);
    const dose = minDose + ((clamped - pad.left) / plotW) * (maxDose - minDose);
    const values = series.map((entry) => {
      const idx = closestIndex(entry.dose, dose);
      const vol =
        entry.volume[idx] ??
        entry.volume[entry.volume.length - 1] ??
        null;
      return {
        name: entry.name,
        color: entry.color,
        volume: vol !== null ? toPercent(vol) : null,
      };
    });
    setHoverX(clamped);
    onHover?.({ dose, values });
  }

  function handleLeave() {
    setHoverX(null);
    onHover?.(null);
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      height="320"
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
      className="dvh-chart"
    >
      <rect
        x={pad.left}
        y={pad.top}
        width={plotW}
        height={plotH}
        fill="#0b1220"
        stroke="rgba(148,163,184,0.25)"
        rx="10"
      />
      {series.map((entry) => {
        const length = Math.min(entry.dose.length, entry.volume.length);
        const points = Array.from({ length }, (_, idx) => {
          const dose = entry.dose[idx];
          const vol = toPercent(entry.volume[idx]);
          const x = pad.left + ((dose - minDose) / (maxDose - minDose || 1)) * plotW;
          const y = pad.top + (1 - (vol - minVol) / (maxVol - minVol || 1)) * plotH;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(" ");
        return (
          <polyline
            key={entry.name}
            points={points}
            fill="none"
            stroke={entry.color}
            strokeWidth="2"
            strokeLinecap="round"
          />
        );
      })}
      {[0, 25, 50, 75, 100].map((tick) => {
        const y = pad.top + (1 - tick / 100) * plotH;
        return (
          <g key={`y-${tick}`}>
            <line
              x1={pad.left}
              x2={pad.left + plotW}
              y1={y}
              y2={y}
              stroke="rgba(148,163,184,0.15)"
            />
            <text x={10} y={y + 4} fontSize="10" fill="#94a3b8">
              {tick}%
            </text>
          </g>
        );
      })}
      {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const tickDose = minDose + ratio * (maxDose - minDose || 1);
        const x = pad.left + ratio * plotW;
        return (
          <text key={`x-${ratio}`} x={x - 6} y={height - 10} fontSize="10" fill="#94a3b8">
            {tickDose.toFixed(0)}
          </text>
        );
      })}
      {hoverX !== null ? (
        <line
          x1={hoverX}
          x2={hoverX}
          y1={pad.top}
          y2={pad.top + plotH}
          stroke="rgba(248,250,252,0.35)"
          strokeDasharray="4 4"
        />
      ) : null}
      <text x={pad.left} y={height - 10} fontSize="11" fill="#94a3b8">
        Dose (Gy)
      </text>
      <text x={8} y={pad.top + 12} fontSize="11" fill="#94a3b8">
        Volume (%)
      </text>
    </svg>
  );
}

export default function HomePage() {
  const [caseId, setCaseId] = useState(CASE_OPTIONS[0].id);
  const [protocol, setProtocol] = useState("Lung_2Gy_30Fx");
  const [preset, setPreset] = useState("super_fast");
  const [availableRuns, setAvailableRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [runId, setRunId] = useState(null);
  const [status, setStatus] = useState(null);
  const [events, setEvents] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [timing, setTiming] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [metricsColumns, setMetricsColumns] = useState([]);
  const [tracePoints, setTracePoints] = useState([]);
  const [dvhImageUrl, setDvhImageUrl] = useState("");
  const [dvhData, setDvhData] = useState(null);
  const [dvhHover, setDvhHover] = useState(null);
  const [ctMeta, setCtMeta] = useState(null);
  const [ctSliceIndex, setCtSliceIndex] = useState(0);
  const [ctWindow, setCtWindow] = useState(400);
  const [ctLevel, setCtLevel] = useState(40);
  const [doseOverlay, setDoseOverlay] = useState(false);
  const [doseOpacity, setDoseOpacity] = useState(0.5);
  const [doseMin, setDoseMin] = useState(0);
  const [doseMax, setDoseMax] = useState(null);
  const [doseMeta, setDoseMeta] = useState(null);
  const [doseStatus, setDoseStatus] = useState("idle");
  const [error, setError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const resultsLoaded = useRef(false);

  const apiBase = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

  async function refreshRuns() {
    try {
      const response = await fetch(`${apiBase}/runs`);
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      const runs = payload.runs || [];
      setAvailableRuns(runs);
      if (!selectedRunId && runs.length) {
        setSelectedRunId(runs[0].run_id);
      }
    } catch (err) {
      setAvailableRuns([]);
    }
  }

  async function startRun() {
    setError(null);
    setEvents([]);
    setTiming(null);
    setMetrics([]);
    setMetricsColumns([]);
    setTracePoints([]);
    setDvhImageUrl("");
    setDvhData(null);
    setDvhHover(null);
    setCtMeta(null);
    setCtSliceIndex(0);
    setDoseOverlay(false);
    setDoseOpacity(0.5);
    setDoseMin(0);
    setDoseMax(null);
    setDoseMeta(null);
    setDoseStatus("idle");
    resultsLoaded.current = false;
    setIsRunning(true);
    try {
      const payload = {
        case_id: caseId,
        protocol,
        adapter: "example",
        fast: preset === "fast",
        super_fast: preset === "super_fast",
      };
      const response = await fetch(`${apiBase}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }
      const data = await response.json();
      setRunId(data.run_id);
      setStatus({ state: "queued" });
      refreshRuns();
    } catch (err) {
      setError(err.message || "Failed to start run.");
      setIsRunning(false);
    }
  }

  function loadRun(runValue) {
    if (!runValue) {
      return;
    }
    setError(null);
    setEvents([]);
    setTiming(null);
    setMetrics([]);
    setMetricsColumns([]);
    setTracePoints([]);
    setDvhImageUrl("");
    setDvhData(null);
    setDvhHover(null);
    setCtMeta(null);
    setCtSliceIndex(0);
    setDoseOverlay(false);
    setDoseOpacity(0.5);
    setDoseMin(0);
    setDoseMax(null);
    setDoseMeta(null);
    setDoseStatus("idle");
    resultsLoaded.current = false;
    setRunId(runValue);
    setIsRunning(false);
  }

  async function fetchCtMeta(runValue) {
    if (!runValue) {
      setCtMeta(null);
      return;
    }
    try {
      const response = await fetch(`${apiBase}/runs/${runValue}/ct/meta`);
      if (!response.ok) {
        setCtMeta(null);
        return;
      }
      const payload = await response.json();
      setCtMeta(payload);
      if (payload.slice_count) {
        setCtSliceIndex(Math.floor(payload.slice_count / 2));
      }
    } catch (err) {
      setCtMeta(null);
    }
  }

  async function createDose3d() {
    if (!runId) {
      return;
    }
    setDoseStatus("creating");
    setError(null);
    try {
      const response = await fetch(`${apiBase}/runs/${runId}/dose-3d`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`Dose export failed: ${response.status}`);
      }
      const payload = await response.json();
      if (payload.status === "created" || payload.status === "exists") {
        setDoseStatus("ready");
      } else {
        setDoseStatus("idle");
      }
    } catch (err) {
      setDoseStatus("error");
      setError(err.message || "Failed to export 3D dose.");
    }
  }

  useEffect(() => {
    refreshRuns();
    const interval = setInterval(refreshRuns, 8000);
    return () => clearInterval(interval);
  }, [apiBase]);

  useEffect(() => {
    if (!runId) {
      return undefined;
    }
    const source = new EventSource(`${apiBase}/runs/${runId}/events`);
    source.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        setEvents((prev) => [...prev.slice(-200), parsed]);
        if (parsed.stage === "trace" && parsed.data) {
          setTracePoints((prev) => [...prev, parsed.data]);
        }
      } catch (err) {
        setEvents((prev) => [...prev, { stage: "log", message: event.data }]);
      }
    };
    source.onerror = () => {
      source.close();
    };
    return () => source.close();
  }, [apiBase, runId]);

  useEffect(() => {
    fetchCtMeta(runId);
  }, [apiBase, runId]);

  useEffect(() => {
    if (!runId) {
      return undefined;
    }
    let active = true;
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${apiBase}/runs/${runId}`);
        if (!response.ok) {
          return;
        }
        const payload = await response.json();
        if (!active) {
          return;
        }
        setStatus(payload.status);
        const nextArtifacts = payload.artifacts || [];
        setArtifacts((prev) => (arraysEqual(prev, nextArtifacts) ? prev : nextArtifacts));
        if (payload.artifacts?.includes("dvh_steps.png")) {
          setDvhImageUrl(`${apiBase}/runs/${runId}/artifacts/dvh_steps.png`);
        }
        if (payload.status?.state === "error") {
          setError(payload.status?.error || "Run failed");
          setIsRunning(false);
        }
        if (payload.status?.state === "completed" && !resultsLoaded.current) {
          resultsLoaded.current = true;
          setIsRunning(false);
          const [timingRes, metricsRes, traceRes] = await Promise.all([
            fetch(`${apiBase}/runs/${runId}/artifacts/timing.json`),
            fetch(`${apiBase}/runs/${runId}/artifacts/metrics.json`),
            fetch(`${apiBase}/runs/${runId}/artifacts/solver_trace.json`),
          ]);
          if (timingRes.ok) {
            setTiming(await timingRes.json());
          }
          if (metricsRes.ok) {
            const metricsPayload = await metricsRes.json();
            setMetrics(metricsPayload.records || metricsPayload.metrics || []);
            setMetricsColumns(metricsPayload.columns || []);
          }
          if (traceRes.ok) {
            setTracePoints(await traceRes.json());
          }
        }
      } catch (err) {
        if (active) {
          setError(err.message || "Failed to fetch run status.");
          setIsRunning(false);
        }
      }
    }, 1400);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [apiBase, runId]);

  useEffect(() => {
    setDoseStatus((prev) => {
      if (artifacts.includes("dose_3d.npy")) {
        return "ready";
      }
      if (prev === "ready") {
        return "idle";
      }
      return prev;
    });
  }, [artifacts]);

  useEffect(() => {
    if (!runId || !artifacts.includes("dvh.json")) {
      setDvhData(null);
      return;
    }
    if (dvhData) {
      return;
    }
    fetch(`${apiBase}/runs/${runId}/artifacts/dvh.json`)
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => setDvhData(payload))
      .catch(() => setDvhData(null));
  }, [apiBase, runId, artifacts, dvhData]);

  const dvhSeries = useMemo(() => {
    if (!dvhData) {
      return [];
    }
    return Object.entries(dvhData)
      .map(([name, values], idx) => {
        const rawDose = Array.isArray(values?.dose_gy) ? values.dose_gy : [];
        const rawVol = Array.isArray(values?.volume_fraction) ? values.volume_fraction : [];
        const dose = rawDose
          .map((value) => (Number.isFinite(value) ? value : Number.parseFloat(value)))
          .filter((value) => Number.isFinite(value));
        const volume = rawVol
          .map((value) => (Number.isFinite(value) ? value : Number.parseFloat(value)))
          .filter((value) => Number.isFinite(value));
        const length = Math.min(dose.length, volume.length);
        const stride = length > 1200 ? Math.ceil(length / 1200) : 1;
        const doseSampled = stride === 1 ? dose : dose.filter((_, i) => i % stride === 0);
        const volSampled = stride === 1 ? volume : volume.filter((_, i) => i % stride === 0);
        return {
          name,
          dose: doseSampled,
          volume: volSampled,
          maxDose: safeArrayMax(doseSampled),
          volumeMax: safeArrayMax(volSampled),
          color: DVH_COLORS[idx % DVH_COLORS.length],
        };
      })
      .filter((entry) => entry.dose.length && entry.volume.length);
  }, [dvhData]);

  useEffect(() => {
    if (!runId || !artifacts.includes("dose_3d_meta.json")) {
      setDoseMeta(null);
      return;
    }
    if (doseMeta) {
      return;
    }
    fetch(`${apiBase}/runs/${runId}/artifacts/dose_3d_meta.json`)
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => setDoseMeta(payload))
      .catch(() => setDoseMeta(null));
  }, [apiBase, runId, artifacts, doseMeta]);

  useEffect(() => {
    if (doseMax !== null) {
      return;
    }
    if (doseMeta?.max_gy) {
      setDoseMax(Math.round(doseMeta.max_gy * 100) / 100);
      return;
    }
    if (dvhSeries.length) {
      const maxDose = safeArrayMax(dvhSeries.map((entry) => entry.maxDose));
      if (Number.isFinite(maxDose) && maxDose > 0) {
        setDoseMax(Math.round(maxDose * 100) / 100);
      }
    }
  }, [doseMeta, doseMax, dvhSeries]);

  const stageStatus = useMemo(() => {
    const seen = new Set(events.map((event) => event.stage));
    return STAGES.map((stage) => ({
      ...stage,
      done: seen.has(stage.id),
    }));
  }, [events]);

  const metricsDisplay = useMemo(() => {
    return metrics.map((row) => {
      const planValue = extractPlanValue(row["Plan Value"]) || row["Plan Value"];
      return {
        ...row,
        "Plan Value": planValue,
        __status: computeStatus(row, planValue),
      };
    });
  }, [metrics]);

  const displayColumns = metricsColumns.length
    ? metricsColumns
    : metricsDisplay.length
      ? Object.keys(metricsDisplay[0])
      : [];

  const lastEvent = events[events.length - 1];
  const sliceCount = ctMeta?.slice_count || 0;
  const sliceMax = sliceCount > 0 ? sliceCount - 1 : 0;
  const doseReady = doseStatus === "ready";
  const ctImageUrl = useMemo(() => {
    if (!runId || !ctMeta) {
      return "";
    }
    return `${apiBase}/runs/${runId}/ct/slice?index=${ctSliceIndex}&window=${ctWindow}&level=${ctLevel}`;
  }, [apiBase, runId, ctMeta, ctSliceIndex, ctWindow, ctLevel]);
  const doseImageUrl = useMemo(() => {
    if (!runId || !doseOverlay || !artifacts.includes("dose_3d.npy")) {
      return "";
    }
    const maxValue = doseMax ?? doseMeta?.max_gy ?? 1;
    const minValue = doseMin ?? 0;
    return `${apiBase}/runs/${runId}/dose/slice?index=${ctSliceIndex}&dose_min=${minValue}&dose_max=${maxValue}`;
  }, [apiBase, runId, doseOverlay, artifacts, ctSliceIndex, doseMin, doseMax, doseMeta]);
  const handleSliceChange = (next) => {
    setCtSliceIndex((prev) => (prev === next ? prev : next));
  };
  const handleSliceWheel = (event) => {
    if (!sliceCount) {
      return;
    }
    event.preventDefault();
    const direction = event.deltaY > 0 ? 1 : -1;
    setCtSliceIndex((prev) => Math.min(sliceMax, Math.max(0, prev + direction)));
  };

  return (
    <div>
      <header className="topbar">
        <div className="brand">
          <img src="/jefferson-university-2.svg" alt="TJU" />
          <div>
            <div className="brand-title">ECHO VMAT Workbench</div>
            <div className="brand-subtitle">Research planning console</div>
          </div>
        </div>
        <div className="toolbar">
          <div className="toolbar-group">
            <label htmlFor="case">Case</label>
            <select id="case" value={caseId} onChange={(event) => setCaseId(event.target.value)}>
              {CASE_OPTIONS.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <span className={`status-pill ${status?.state === "error" ? "error" : ""}`}>
            {status?.state || "idle"}
          </span>
          <button className="btn btn-ghost" onClick={() => loadRun(selectedRunId)}>
            Load Run
          </button>
          <button className="btn btn-primary" onClick={startRun} disabled={isRunning}>
            {isRunning ? "Running" : "Run Example"}
          </button>
        </div>
      </header>

      <main className="dashboard">
        <section className="panel">
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">CT Viewer (Axial)</div>
                <div className="card-subtitle">Slice navigation + window/level</div>
              </div>
              <button
                className="btn"
                onClick={createDose3d}
                disabled={!runId || doseReady || doseStatus === "creating"}
              >
                {doseReady
                  ? "3D Dose Ready"
                  : doseStatus === "creating"
                    ? "Creating Dose..."
                    : "Create 3D Dose"}
              </button>
            </div>
            {ctMeta ? (
              <div className="ct-viewer">
                <div className="ct-canvas" onWheel={handleSliceWheel}>
                  {ctImageUrl ? (
                    <img
                      key={`${ctSliceIndex}-${ctWindow}-${ctLevel}`}
                      className="ct-image"
                      src={ctImageUrl}
                      alt="CT slice"
                      draggable={false}
                    />
                  ) : (
                    <div className="placeholder">CT slice not available.</div>
                  )}
                  {doseOverlay && doseImageUrl ? (
                    <img
                      className="ct-dose"
                      src={doseImageUrl}
                      alt="Dose overlay"
                      style={{ opacity: doseOpacity }}
                      draggable={false}
                    />
                  ) : null}
                </div>
                <div className="ct-controls">
                  <div className="ct-control">
                    <label htmlFor="ct-slice">Slice {sliceCount ? ctSliceIndex + 1 : 0} / {sliceCount || "--"}</label>
                    <input
                      id="ct-slice"
                      type="range"
                      min="0"
                      max={sliceMax}
                      value={ctSliceIndex}
                      onChange={(event) => handleSliceChange(Number(event.target.value))}
                      disabled={!sliceCount}
                    />
                  </div>
                  <div className="ct-control-row">
                    <div className="ct-control">
                      <label htmlFor="ct-window">Window</label>
                      <input
                        id="ct-window"
                        type="number"
                        value={ctWindow}
                        min="1"
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          if (Number.isFinite(value) && value > 0) {
                            setCtWindow(value);
                          }
                        }}
                      />
                    </div>
                    <div className="ct-control">
                      <label htmlFor="ct-level">Level</label>
                      <input
                        id="ct-level"
                        type="number"
                        value={ctLevel}
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          if (Number.isFinite(value)) {
                            setCtLevel(value);
                          }
                        }}
                      />
                    </div>
                  </div>
                  <div className="ct-control-row">
                    <div className="ct-control">
                      <label htmlFor="dose-overlay">Dose Overlay</label>
                      <select
                        id="dose-overlay"
                        value={doseOverlay ? "on" : "off"}
                        onChange={(event) => setDoseOverlay(event.target.value === "on")}
                        disabled={!artifacts.includes("dose_3d.npy")}
                      >
                        <option value="off">Off</option>
                        <option value="on">On</option>
                      </select>
                    </div>
                    <div className="ct-control">
                      <label htmlFor="dose-opacity">Opacity</label>
                      <input
                        id="dose-opacity"
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={doseOpacity}
                        onChange={(event) => setDoseOpacity(Number(event.target.value))}
                        disabled={!doseOverlay}
                      />
                    </div>
                  </div>
                  <div className="ct-control-row">
                    <div className="ct-control">
                      <label htmlFor="dose-max">Dose Max (Gy)</label>
                      <input
                        id="dose-max"
                        type="number"
                        value={doseMax ?? ""}
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          if (Number.isFinite(value) && value > 0) {
                            setDoseMax(value);
                          }
                        }}
                        disabled={!doseOverlay}
                      />
                    </div>
                    <div className="ct-control">
                      <label htmlFor="dose-min">Dose Min (Gy)</label>
                      <input
                        id="dose-min"
                        type="number"
                        value={doseMin}
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          if (Number.isFinite(value) && value >= 0) {
                            setDoseMin(value);
                          }
                        }}
                        disabled={!doseOverlay}
                      />
                    </div>
                  </div>
                  <div className="ct-meta">
                    <span>Spacing: {ctMeta?.resolution_xyz_mm?.join(", ") || "--"} mm</span>
                    <span>Origin: {ctMeta?.origin_xyz_mm?.join(", ") || "--"} mm</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="placeholder">Select a run to load CT data.</div>
            )}
            {doseStatus === "error" ? (
              <div className="placeholder">3D dose export failed. Check backend logs.</div>
            ) : null}
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Run Queue</div>
                <div className="card-subtitle">Latest runs (newest first)</div>
              </div>
            </div>
            <div className="run-list">
              {availableRuns.length ? (
                availableRuns.map((run) => (
                  <div
                    key={run.run_id}
                    className={`run-item ${runId === run.run_id ? "active" : ""}`}
                    onClick={() => loadRun(run.run_id)}
                  >
                    <div className="run-id">{run.run_id}</div>
                    <div className="run-meta">
                      <span>{run.status?.case_id || "unknown"}</span>
                      <span>{run.status?.state || "unknown"}</span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="placeholder">No runs yet.</div>
              )}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Run Setup</div>
                <div className="card-subtitle">Lightweight presets only</div>
              </div>
            </div>
            <div style={{ display: "grid", gap: "10px" }}>
              <label htmlFor="protocol">Protocol</label>
              <input
                id="protocol"
                value={protocol}
                onChange={(event) => setProtocol(event.target.value)}
              />
              <label htmlFor="preset">Preset</label>
              <select id="preset" value={preset} onChange={(event) => setPreset(event.target.value)}>
                {PRESETS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
              <label htmlFor="load-run">Load by ID</label>
              <select
                id="load-run"
                value={selectedRunId}
                onChange={(event) => setSelectedRunId(event.target.value)}
              >
                <option value="">Select a run</option>
                {availableRuns.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.run_id} ({run.status?.state || "unknown"})
                  </option>
                ))}
              </select>
              <button className="btn" onClick={() => loadRun(selectedRunId)} disabled={!selectedRunId}>
                Load Selected
              </button>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Status Timeline</div>
                <div className="card-subtitle">Live pipeline events</div>
              </div>
            </div>
            <ul className="progress-list">
              {stageStatus.map((stage) => (
                <li key={stage.id} className={`progress-item ${stage.done ? "done" : ""}`}>
                  <span>{stage.label}</span>
                  <span>{stage.done ? "done" : "pending"}</span>
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section className="panel">
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Dose Volume Histogram</div>
                <div className="card-subtitle">Interactive DVH (hover for values)</div>
              </div>
            </div>
            <DvhPlot series={dvhSeries} onHover={setDvhHover} />
            {dvhHover ? (
              <div className="dvh-hover">
                <div>Hover dose: {dvhHover.dose.toFixed(2)} Gy</div>
                <div className="dvh-hover-grid">
                  {dvhHover.values.map((entry) => (
                    <div key={entry.name} className="dvh-hover-item">
                      <span className="dvh-swatch" style={{ background: entry.color }} />
                      <span>{entry.name}</span>
                      <span>{entry.volume !== null ? `${entry.volume.toFixed(1)}%` : "--"}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : dvhImageUrl ? (
              <div className="card-subtitle">Static snapshot available in artifacts.</div>
            ) : null}
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Optimization Trace</div>
                <div className="card-subtitle">Objective trend (live)</div>
              </div>
            </div>
            <div className="chart-wrap">
              <TracePlot points={tracePoints} />
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Timing Summary</div>
                <div className="card-subtitle">Wall-clock checkpoints</div>
              </div>
            </div>
            <div className="kpi-grid">
              {TIMING_LABELS.map((item) => (
                <div className="kpi" key={item.key}>
                  <div className="label">{item.label}</div>
                  <div className="value">{formatSeconds(timing?.[item.key])}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Live Console</div>
                <div className="card-subtitle">Solver output + system logs</div>
              </div>
            </div>
            <div className="console">
              {events.length === 0 ? (
                <div className="placeholder">Waiting for output.</div>
              ) : (
                events.map((event, idx) => (
                  <div key={`${event.ts || idx}`} className="console-line">
                    {event.ts ? `${event.ts} ` : ""}
                    {event.stage ? `[${event.stage}] ` : ""}
                    {event.message || ""}
                  </div>
                ))
              )}
            </div>
            {lastEvent ? (
              <div className="card-subtitle" style={{ marginTop: "10px" }}>
                Latest: {lastEvent.message}
              </div>
            ) : null}
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Clinical Criteria</div>
                <div className="card-subtitle">Plan values vs limits</div>
              </div>
            </div>
            {error ? <div className="placeholder">{error}</div> : null}
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {displayColumns.length ? (
                      displayColumns.map((col) => <th key={col}>{col}</th>)
                    ) : (
                      <th>Metric</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {metricsDisplay.length ? (
                    metricsDisplay.map((row, idx) => (
                      <tr key={`row-${idx}`} className={`row-${row.__status || "neutral"}`}>
                        {displayColumns.map((col) => (
                          <td key={`${idx}-${col}`}>{row?.[col] ?? "--"}</td>
                        ))}
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={displayColumns.length || 1} className="placeholder">
                        No metrics available yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            {artifacts.length ? (
              <div className="card-subtitle" style={{ marginTop: "10px" }}>
                Artifacts: {artifacts.join(", ")}
              </div>
            ) : null}
          </div>
        </section>
      </main>
    </div>
  );
}
