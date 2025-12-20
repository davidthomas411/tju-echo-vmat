"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const CASE_OPTIONS = [
  { id: "Lung_Patient_11", label: "Lung Patient 11" },
  { id: "Lung_Patient_4", label: "Lung Patient 4" },
  { id: "Lung_Patient_2", label: "Lung Patient 2" },
  { id: "Lung_Phantom_Patient_1", label: "Lung Phantom Patient 1" },
];

const PRESETS = [
  {
    id: "super_fast",
    label: "Super Fast",
    detail: "Exploratory run with sparse DDC and fewer iterations.",
  },
  {
    id: "fast",
    label: "Fast",
    detail: "Planner beams + sparse DDC with full correction loop.",
  },
  {
    id: "balanced",
    label: "Balanced",
    detail: "Baseline example settings from the repo.",
  },
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
  { key: "dataset_download_sec", label: "Dataset download" },
  { key: "case_load_sec", label: "Case load" },
  { key: "ddc_load_sec", label: "DDC" },
  { key: "optimization_sec", label: "Optimization" },
  { key: "evaluation_sec", label: "Evaluation" },
  { key: "total_sec", label: "Total" },
];

const palette = [
  "#b21e2b",
  "#1f4b7a",
  "#0f6b6f",
  "#2f9e44",
  "#8a3ffc",
  "#a63d40",
  "#3c5a82",
  "#6b705c",
  "#8c5e3c",
];

function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return `${value.toFixed(2)} s`;
}

function renderCell(value) {
  if (value === null || value === undefined || value === "") {
    return "--";
  }
  if (typeof value === "string" && value.includes("<div")) {
    return <span dangerouslySetInnerHTML={{ __html: value }} />;
  }
  return String(value);
}

function DvhPlot({ dvh }) {
  if (!dvh || Object.keys(dvh).length === 0) {
    return <div className="muted">No DVH available yet.</div>;
  }

  const entries = Object.entries(dvh);
  const maxDose = Math.max(
    1,
    ...entries.flatMap(([, data]) => (data.dose_gy || []).slice(-1))
  );

  const width = 520;
  const height = 260;
  const pad = { left: 46, right: 16, top: 18, bottom: 32 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const xTicks = 5;
  const yTicks = 4;

  return (
    <div className="dvh-card">
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="260">
        <rect
          x={pad.left}
          y={pad.top}
          width={plotW}
          height={plotH}
          fill="#ffffff"
          stroke="#d7dde5"
          strokeWidth="1"
          rx="10"
        />
        {Array.from({ length: xTicks }).map((_, idx) => {
          const x = pad.left + (plotW / (xTicks - 1)) * idx;
          return (
            <g key={`x-${idx}`}>
              <line
                x1={x}
                y1={pad.top}
                x2={x}
                y2={pad.top + plotH}
                stroke="#eef1f5"
              />
              <text x={x} y={height - 8} fontSize="10" textAnchor="middle" fill="#5b6b7a">
                {Math.round((maxDose / (xTicks - 1)) * idx)}
              </text>
            </g>
          );
        })}
        {Array.from({ length: yTicks }).map((_, idx) => {
          const y = pad.top + (plotH / (yTicks - 1)) * idx;
          const value = 100 - (100 / (yTicks - 1)) * idx;
          return (
            <g key={`y-${idx}`}>
              <line
                x1={pad.left}
                y1={y}
                x2={pad.left + plotW}
                y2={y}
                stroke="#eef1f5"
              />
              <text x={8} y={y + 4} fontSize="10" fill="#5b6b7a">
                {value}%
              </text>
            </g>
          );
        })}
        {entries.map(([name, data], idx) => {
          const dose = data.dose_gy || [];
          const vol = data.volume_fraction || [];
          const points = dose
            .map((x, i) => {
              const y = vol[i] ?? 0;
              const px = pad.left + (x / maxDose) * plotW;
              const py = pad.top + (1 - Math.min(Math.max(y, 0), 1)) * plotH;
              return `${px.toFixed(1)},${py.toFixed(1)}`;
            })
            .join(" ");
          return (
            <polyline
              key={name}
              fill="none"
              stroke={palette[idx % palette.length]}
              strokeWidth="2"
              points={points}
              opacity="0.9"
            />
          );
        })}
      </svg>
      <div className="dvh-legend">
        {entries.map(([name], idx) => (
          <span key={name}>
            <span className="legend-dot" style={{ background: palette[idx % palette.length] }} />
            {name}
          </span>
        ))}
      </div>
    </div>
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
  const [dvh, setDvh] = useState(null);
  const [dvhImageUrl, setDvhImageUrl] = useState("");
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

  const stageStatus = useMemo(() => {
    const seen = new Set(events.map((event) => event.stage));
    return STAGES.map((stage) => ({
      ...stage,
      done: seen.has(stage.id),
    }));
  }, [events]);

  const lastEvent = events[events.length - 1];

  async function startRun() {
    setError(null);
    setEvents([]);
    setTiming(null);
    setMetrics([]);
    setMetricsColumns([]);
    setDvh(null);
    setDvhImageUrl("");
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

  async function loadRun() {
    if (!selectedRunId) {
      return;
    }
    setError(null);
    setEvents([]);
    setTiming(null);
    setMetrics([]);
    setMetricsColumns([]);
    setDvh(null);
    setDvhImageUrl("");
    resultsLoaded.current = false;
    setRunId(selectedRunId);
    setIsRunning(false);
  }

  useEffect(() => {
    refreshRuns();
  }, [apiBase]);

  useEffect(() => {
    if (!runId) {
      return undefined;
    }
    const source = new EventSource(`${apiBase}/runs/${runId}/events`);
    source.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        setEvents((prev) => [...prev.slice(-180), parsed]);
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
        setArtifacts(payload.artifacts || []);
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
          const [timingRes, metricsRes] = await Promise.all([
            fetch(`${apiBase}/runs/${runId}/artifacts/timing.json`),
            fetch(`${apiBase}/runs/${runId}/artifacts/metrics.json`),
          ]);
          if (timingRes.ok) {
            setTiming(await timingRes.json());
          }
          if (metricsRes.ok) {
            const metricsPayload = await metricsRes.json();
            setMetrics(metricsPayload.records || metricsPayload.metrics || []);
            setMetricsColumns(metricsPayload.columns || []);
          }
        }
      } catch (err) {
        if (active) {
          setError(err.message || "Failed to fetch run status.");
          setIsRunning(false);
        }
      }
    }, 1200);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [apiBase, runId]);

  return (
    <div className="page">
      <header className="header">
        <img src="/jefferson-university-2.svg" alt="Thomas Jefferson University" />
        <div>
          <div className="header-eyebrow">Thomas Jefferson University</div>
          <div className="header-title">ECHO Workbench</div>
          <p className="header-subtitle">
            Local VMAT research console for running ECHO-VMAT examples, tracking solver
            timing, and reviewing DVH-driven plan quality.
          </p>
        </div>
      </header>

      <main className="grid">
        <section className="card" style={{ animationDelay: "0.05s" }}>
          <h2>Run Setup</h2>
          <div className="form-grid">
            <div>
              <div className="label">Case ID</div>
              <select value={caseId} onChange={(event) => setCaseId(event.target.value)}>
                {CASE_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div className="label">Protocol</div>
              <input
                value={protocol}
                onChange={(event) => setProtocol(event.target.value)}
                placeholder="Lung_2Gy_30Fx"
              />
            </div>
            <div>
              <div className="label">Preset</div>
              <div className="radio-row">
                {PRESETS.map((option) => (
                  <label key={option.id} className="radio-pill">
                    <input
                      type="radio"
                      name="preset"
                      value={option.id}
                      checked={preset === option.id}
                      onChange={() => setPreset(option.id)}
                    />
                    <div>
                      <div>{option.label}</div>
                      <div className="muted">{option.detail}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>
            <div>
              <button className="button" onClick={startRun} disabled={isRunning}>
                {isRunning ? "Run in progress" : "Run example"}
              </button>
            </div>
            <div>
              <div className="label">Load Existing Run</div>
              <input
                list="run-options"
                value={selectedRunId}
                onChange={(event) => setSelectedRunId(event.target.value)}
                placeholder="Select or paste run_id"
              />
              <datalist id="run-options">
                {availableRuns.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.run_id} ({run.status?.state || "unknown"})
                  </option>
                ))}
              </datalist>
              <div style={{ marginTop: "10px" }}>
                <button className="button" onClick={loadRun} disabled={!selectedRunId}>
                  Load run
                </button>
              </div>
            </div>
          </div>
          {runId ? (
            <div className="muted" style={{ marginTop: "12px" }}>
              Active run: {runId}
            </div>
          ) : null}
        </section>

        <section className="stack">
          <div className="card" style={{ animationDelay: "0.1s" }}>
            <h2>Live Progress</h2>
            <div className="status-row">
              <span className={`badge ${status?.state === "error" ? "error" : ""}`}>
                {status?.state || "idle"}
              </span>
              {lastEvent ? <span className="muted">{lastEvent.message}</span> : null}
            </div>
            <ul className="progress-list">
              {stageStatus.map((stage) => (
                <li
                  key={stage.id}
                  className={`progress-item ${stage.done ? "done" : ""}`}
                >
                  <span>{stage.label}</span>
                  <span className="muted">{stage.done ? "done" : "pending"}</span>
                </li>
              ))}
            </ul>
            <div className="log-box">
              {events.length === 0 ? (
                <div className="muted">Waiting for solver output.</div>
              ) : (
                events.map((event, idx) => (
                  <div key={`${event.ts}-${idx}`} className="log-line">
                    {event.ts ? `${event.ts} ` : ""}
                    {event.stage ? `[${event.stage}] ` : ""}
                    {event.message || event}
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="card" style={{ animationDelay: "0.15s" }}>
            <h2>Results</h2>
            {error ? <div className="muted">{error}</div> : null}
            {!error && !timing && !dvh ? (
              <div className="muted">Results will appear once the run finishes.</div>
            ) : (
              <div className="results-grid">
                <div>
                  <div className="label" style={{ marginBottom: "8px" }}>
                    Timing
                  </div>
                  <ul className="timing-list">
                    {TIMING_LABELS.map((row) => (
                      <li key={row.key} className="timing-item">
                        <span>{row.label}</span>
                        <span>{formatSeconds(timing?.[row.key])}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <div className="label" style={{ marginBottom: "8px" }}>
                    DVH
                  </div>
                  {dvhImageUrl ? (
                    <img className="dvh-image" src={dvhImageUrl} alt="DVH plot" />
                  ) : (
                    <DvhPlot dvh={dvh} />
                  )}
                </div>

                <div>
                  <div className="label" style={{ marginBottom: "8px" }}>
                    Metrics
                  </div>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          {(metricsColumns.length ? metricsColumns : ["Metric"]).map((col) => (
                            <th key={col}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(metrics || []).slice(0, 50).map((row, idx) => (
                          <tr key={`row-${idx}`}>
                            {(metricsColumns.length ? metricsColumns : Object.keys(row || {})).map((col) => (
                              <td key={`${idx}-${col}`}>{renderCell(row?.[col])}</td>
                            ))}
                          </tr>
                        ))}
                        {metrics.length === 0 ? (
                          <tr>
                            <td colSpan={metricsColumns.length || 1} className="muted">
                              No metrics available yet.
                            </td>
                          </tr>
                        ) : null}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
