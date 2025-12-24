"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const PRESETS = [
  { id: "super_fast", label: "Super Fast" },
  { id: "fast", label: "Safe (low memory)" },
  { id: "balanced", label: "Balanced" },
];

const OPTIMIZERS = [
  { id: "echo-vmat", label: "ECHO-VMAT" },
  { id: "compressrtp", label: "CompressRTP" },
];

const COMPRESS_MODES = [
  { id: "sparse-only", label: "Sparse Only (RMR)" },
  { id: "sparse-plus-low-rank", label: "Sparse + Low Rank" },
  { id: "wavelet", label: "Wavelet Smoothness" },
];

const COMPRESS_STEPS = [
  { id: "all", label: "Full Pipeline" },
  { id: "ddc", label: "DDC Only" },
  { id: "sparse", label: "Sparse Only" },
  { id: "svd", label: "Sparse + Low Rank" },
  { id: "wavelet", label: "Wavelet Basis" },
];

const STAGES = [
  { id: "dataset_download", label: "Dataset" },
  { id: "case_load", label: "Case Load" },
  { id: "beams", label: "Beam Setup" },
  { id: "ddc", label: "DDC" },
  { id: "compress", label: "Compression" },
  { id: "optimization", label: "Optimization" },
  { id: "evaluation", label: "Evaluation" },
  { id: "complete", label: "Complete" },
];

const TIMING_LABELS = [
  { key: "dataset_download_sec", label: "Dataset" },
  { key: "case_load_sec", label: "Case Load" },
  { key: "beams_sec", label: "Beams" },
  { key: "ddc_load_sec", label: "DDC" },
  { key: "compress_sec", label: "Compression" },
  { key: "optimization_sec", label: "Optimization" },
  { key: "evaluation_sec", label: "Evaluation" },
  { key: "total_sec", label: "Total" },
];

const RESOURCE_LABELS = [
  { key: "max_rss_mb", label: "Peak RSS (MB)" },
  { key: "case_load_rss_mb", label: "Case Load RSS (MB)" },
  { key: "beams_rss_mb", label: "Beams RSS (MB)" },
  { key: "ddc_rss_mb", label: "DDC RSS (MB)" },
  { key: "compress_rss_mb", label: "Compression RSS (MB)" },
  { key: "optimization_rss_mb", label: "Optimization RSS (MB)" },
  { key: "evaluation_rss_mb", label: "Evaluation RSS (MB)" },
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

function formatDuration(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  const total = Math.max(0, Math.round(value));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    const rem = minutes % 60;
    return `${hours}h ${rem}m ${seconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return Number(value).toFixed(1);
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

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return `${value.toFixed(1)}%`;
}

function percentileColor(value) {
  const clamped = Math.min(100, Math.max(0, Number(value) || 0));
  const hue = (clamped / 100) * 120;
  return `hsl(${hue}, 70%, 50%)`;
}

function buildHistogram(values, binCount) {
  if (!values.length || binCount <= 0) {
    return { bins: [], max: 0, min: 0 };
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const bins = Array.from({ length: binCount }, () => 0);
  values.forEach((value) => {
    const idx = Math.min(binCount - 1, Math.floor(((value - min) / span) * binCount));
    bins[idx] += 1;
  });
  return { bins, min, max, maxCount: Math.max(...bins) };
}

function polarPoint(cx, cy, radius, angle) {
  return {
    x: cx + radius * Math.cos(angle),
    y: cy + radius * Math.sin(angle),
  };
}

function arcPath(cx, cy, innerR, outerR, startAngle, endAngle) {
  const startOuter = polarPoint(cx, cy, outerR, endAngle);
  const endOuter = polarPoint(cx, cy, outerR, startAngle);
  const startInner = polarPoint(cx, cy, innerR, startAngle);
  const endInner = polarPoint(cx, cy, innerR, endAngle);
  const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;
  return [
    `M ${startOuter.x} ${startOuter.y}`,
    `A ${outerR} ${outerR} 0 ${largeArc} 0 ${endOuter.x} ${endOuter.y}`,
    `L ${startInner.x} ${startInner.y}`,
    `A ${innerR} ${innerR} 0 ${largeArc} 1 ${endInner.x} ${endInner.y}`,
    "Z",
  ].join(" ");
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

function buildDvhSeries(data, label, colorMap, dash) {
  if (!data) {
    return [];
  }
  return Object.entries(data)
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
        key: `${label}-${name}-${idx}`,
        name: label ? `${label} ${name}` : name,
        structName: name,
        group: label || "Current",
        dose: doseSampled,
        volume: volSampled,
        maxDose: safeArrayMax(doseSampled),
        volumeMax: safeArrayMax(volSampled),
        color: colorMap.get(name) || DVH_COLORS[idx % DVH_COLORS.length],
        dash,
      };
    })
    .filter((entry) => entry.dose.length && entry.volume.length);
}

function metricsToMap(rows) {
  const map = new Map();
  for (const row of rows || []) {
    const structure =
      row["Structure Name"] || row.Structure || row.StructureName || "Unknown";
    const constraint = row.Constraint || row.Metric || "Metric";
    const rawValue = row["Plan Value"] ?? row.PlanValue ?? row.Value;
    const planValue = parseNumeric(extractPlanValue(rawValue) || rawValue);
    map.set(`${structure}|${constraint}`, {
      structure,
      constraint,
      planValue,
    });
  }
  return map;
}

function buildDeltaRows(rowsA, rowsB) {
  const mapA = metricsToMap(rowsA);
  const mapB = metricsToMap(rowsB);
  const keys = new Set([...mapA.keys(), ...mapB.keys()]);
  return Array.from(keys)
    .map((key) => {
      const a = mapA.get(key);
      const b = mapB.get(key);
      const planA = a?.planValue ?? null;
      const planB = b?.planValue ?? null;
      return {
        structure: a?.structure || b?.structure || "Unknown",
        constraint: a?.constraint || b?.constraint || "Metric",
        planA,
        planB,
        delta: planA !== null && planB !== null ? planA - planB : null,
      };
    })
    .sort((left, right) => left.structure.localeCompare(right.structure));
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
  const [hoverX, setHoverX] = useState(null);
  const hoverSeries = useMemo(() => series.filter((entry) => !entry.dimmed), [series]);
  const hoverFrame = useRef(null);
  const hoverPending = useRef(null);

  if (!series.length) {
    return <div className="placeholder">No DVH available yet.</div>;
  }
  const width = 560;
  const height = 320;
  const pad = { left: 54, right: 20, top: 20, bottom: 36 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxDose = Math.max(...series.map((entry) => entry.maxDose || 0)) || 1;
  const minDose = 0;
  const volumeScale = series.some((entry) => entry.volumeMax > 1.5) ? 1 : 100;
  const minVol = 0;
  const maxVol = 100;
  const toPercent = (value) => (volumeScale === 1 ? value : value * 100);

  function handleMove(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const clamped = Math.min(Math.max(x, pad.left), pad.left + plotW);
    hoverPending.current = clamped;
    if (hoverFrame.current) {
      return;
    }
    hoverFrame.current = requestAnimationFrame(() => {
      hoverFrame.current = null;
      const pendingX = hoverPending.current;
      if (pendingX === null) {
        return;
      }
      const dose = minDose + ((pendingX - pad.left) / plotW) * (maxDose - minDose);
      const values = hoverSeries.map((entry) => {
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
      setHoverX(pendingX);
      onHover?.({ dose, values });
    });
  }

  function handleLeave() {
    if (hoverFrame.current) {
      cancelAnimationFrame(hoverFrame.current);
      hoverFrame.current = null;
    }
    hoverPending.current = null;
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
            strokeWidth={entry.dimmed ? "1.4" : "2.2"}
            strokeOpacity={entry.dimmed ? "0.25" : "1"}
            strokeLinecap="round"
            strokeDasharray={entry.dash ? "6 4" : undefined}
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

function DaisyPlot({ score, objectives }) {
  if (!objectives || !objectives.length) {
    return <div className="placeholder">Plan score not available yet.</div>;
  }
  const size = 360;
  const center = size / 2;
  const inner = 58;
  const outer = 150;
  const slice = (Math.PI * 2) / objectives.length;

  return (
    <svg viewBox={`0 0 ${size} ${size}`} width="100%" height="360" className="daisy">
      <circle cx={center} cy={center} r={inner} fill="rgba(15,23,42,0.7)" stroke="rgba(148,163,184,0.3)" />
      {objectives.map((obj, idx) => {
        const baseAngle = -Math.PI / 2 + idx * slice;
        const endAngle = baseAngle + slice * 0.92;
        const pct = obj?.percentile ?? 0;
        const radius = inner + ((outer - inner) * Math.max(0, Math.min(100, pct))) / 100;
        const path = arcPath(center, center, inner, radius, baseAngle, endAngle);
        const fill = percentileColor(pct);
        const opacity = obj.priority === 1 ? 0.85 : obj.priority === 2 ? 0.65 : 0.45;
        const mid = baseAngle + (endAngle - baseAngle) / 2;
        const labelPoint = polarPoint(center, center, outer + 16, mid);
        const label = obj.label?.replace(/\s*\(.*\)/, "") || obj.structure || `Obj ${idx + 1}`;
        const targetPct = obj.target_percentile;
        const targetRadius =
          targetPct !== null && targetPct !== undefined
            ? inner + ((outer - inner) * Math.max(0, Math.min(100, targetPct))) / 100
            : null;
        const targetPoint = targetRadius ? polarPoint(center, center, targetRadius, mid) : null;
        return (
          <g key={obj.id || idx}>
            <path d={path} fill={fill} fillOpacity={opacity} stroke="rgba(148,163,184,0.35)" />
            {targetPoint ? (
              <circle cx={targetPoint.x} cy={targetPoint.y} r="3.5" fill="#f8fafc" stroke="#0f172a" />
            ) : null}
            <text
              x={labelPoint.x}
              y={labelPoint.y}
              fontSize="10"
              fill="#e2e8f0"
              textAnchor={labelPoint.x < center ? "end" : "start"}
            >
              {label}
            </text>
          </g>
        );
      })}
      <text x={center} y={center - 6} textAnchor="middle" fontSize="12" fill="#94a3b8">
        Plan Score
      </text>
      <text x={center} y={center + 18} textAnchor="middle" fontSize="24" fill="#e2e8f0">
        {score?.plan_score ? score.plan_score.toFixed(1) : "--"}
      </text>
      <text x={center} y={center + 36} textAnchor="middle" fontSize="11" fill="#94a3b8">
        {score?.plan_percentile ? `${score.plan_percentile.toFixed(1)}%ile` : "--"}
      </text>
    </svg>
  );
}

function PopulationHistogram({ values, marker }) {
  const width = 560;
  const height = 160;
  const padding = 24;
  const { bins, min, max, maxCount } = buildHistogram(values, 12);
  const barWidth = bins.length ? (width - padding * 2) / bins.length : 0;
  const markerX =
    marker !== null && marker !== undefined && max > min
      ? padding + ((marker - min) / (max - min)) * (width - padding * 2)
      : null;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="population-histogram">
      <rect x="0" y="0" width={width} height={height} rx="12" fill="rgba(15, 23, 42, 0.6)" />
      {bins.map((count, idx) => {
        const barHeight = maxCount ? (count / maxCount) * (height - padding * 2) : 0;
        const x = padding + idx * barWidth;
        const y = height - padding - barHeight;
        return (
          <rect
            key={`bin-${idx}`}
            x={x + 1}
            y={y}
            width={barWidth - 2}
            height={barHeight}
            rx="4"
            fill="rgba(96, 165, 250, 0.8)"
          />
        );
      })}
      {markerX !== null ? (
        <line x1={markerX} x2={markerX} y1={padding} y2={height - padding} stroke="#f97316" strokeWidth="2" />
      ) : null}
      <text x={padding} y={height - 6} fill="#94a3b8" fontSize="11">
        {min.toFixed(1)}
      </text>
      <text x={width - padding} y={height - 6} fill="#94a3b8" fontSize="11" textAnchor="end">
        {max.toFixed(1)}
      </text>
    </svg>
  );
}

export default function HomePage() {
  const [caseOptions, setCaseOptions] = useState([]);
  const [caseFilter, setCaseFilter] = useState("");
  const [caseId, setCaseId] = useState("");
  const [protocol, setProtocol] = useState("Lung_2Gy_30Fx");
  const [activeTab, setActiveTab] = useState("workbench");
  const [preset, setPreset] = useState("super_fast");
  const [optimizer, setOptimizer] = useState("echo-vmat");
  const [compressMode, setCompressMode] = useState("sparse-only");
  const [compressStep, setCompressStep] = useState("all");
  const [thresholdPerc, setThresholdPerc] = useState(10);
  const [rank, setRank] = useState(5);
  const [beamCount, setBeamCount] = useState("");
  const [useGpu, setUseGpu] = useState(false);
  const [tag, setTag] = useState("");
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
  const [planScore, setPlanScore] = useState(null);
  const [planScoreStatus, setPlanScoreStatus] = useState("idle");
  const [populationScores, setPopulationScores] = useState([]);
  const [populationStats, setPopulationStats] = useState(null);
  const [populationStatus, setPopulationStatus] = useState("idle");
  const [populationFilter, setPopulationFilter] = useState("");
  const [compareRunA, setCompareRunA] = useState("");
  const [compareRunB, setCompareRunB] = useState("");
  const [compareDvhA, setCompareDvhA] = useState(null);
  const [compareDvhB, setCompareDvhB] = useState(null);
  const [compareMetricsA, setCompareMetricsA] = useState([]);
  const [compareMetricsB, setCompareMetricsB] = useState([]);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState(null);
  const [ctMeta, setCtMeta] = useState(null);
  const [ctSliceIndex, setCtSliceIndex] = useState(0);
  const [ctWindow, setCtWindow] = useState(400);
  const [ctLevel, setCtLevel] = useState(40);
  const [structures, setStructures] = useState([]);
  const [structureOverlay, setStructureOverlay] = useState(false);
  const [structureSelection, setStructureSelection] = useState("all");
  const [doseOverlay, setDoseOverlay] = useState(false);
  const [doseOpacity, setDoseOpacity] = useState(0.5);
  const [doseMin, setDoseMin] = useState(0);
  const [doseMax, setDoseMax] = useState(null);
  const [doseMeta, setDoseMeta] = useState(null);
  const [doseStatus, setDoseStatus] = useState("idle");
  const [rtPlanStatus, setRtPlanStatus] = useState("idle");
  const [rtPlanFile, setRtPlanFile] = useState("");
  const [rtDoseFile, setRtDoseFile] = useState("");
  const [ctDicomStatus, setCtDicomStatus] = useState("idle");
  const [ctDicomDir, setCtDicomDir] = useState("");
  const [rtStructStatus, setRtStructStatus] = useState("idle");
  const [rtStructPath, setRtStructPath] = useState("");
  const [error, setError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const resultsLoaded = useRef(false);
  const wheelThrottle = useRef(0);

  const apiBase = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

  useEffect(() => {
    let isMounted = true;
    fetch(`${apiBase}/patients?protocol=${encodeURIComponent(protocol)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => {
        if (!isMounted || !payload) {
          return;
        }
        const patients = payload.patients || [];
        const options = patients.map((id) => ({
          id,
          label: id.replaceAll("_", " "),
        }));
        setCaseOptions(options);
        if (!caseId || !patients.includes(caseId)) {
          setCaseId(patients[0] || "");
        }
      })
      .catch(() => {
        if (isMounted) {
          setCaseOptions([]);
        }
      });
    return () => {
      isMounted = false;
    };
  }, [apiBase, protocol]);

  useEffect(() => {
    if (activeTab !== "population") {
      return;
    }
    setPopulationStatus("loading");
    fetch(`${apiBase}/plan-score/population?protocol=${encodeURIComponent(protocol)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => {
        if (!payload) {
          setPopulationScores([]);
          setPopulationStats(null);
          setPopulationStatus("error");
          return;
        }
        setPopulationScores(payload.patients || []);
        setPopulationStats(payload.stats || null);
        setPopulationStatus("ready");
      })
      .catch(() => {
        setPopulationScores([]);
        setPopulationStats(null);
        setPopulationStatus("error");
      });
  }, [apiBase, protocol, activeTab]);

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

  const filteredCaseOptions = useMemo(() => {
    const query = caseFilter.trim().toLowerCase();
    let filtered = caseOptions;
    if (query) {
      filtered = caseOptions.filter((option) => option.label.toLowerCase().includes(query));
    }
    if (caseId && !filtered.some((option) => option.id === caseId)) {
      const selected = caseOptions.find((option) => option.id === caseId);
      if (selected) {
        return [selected, ...filtered];
      }
    }
    return filtered;
  }, [caseOptions, caseFilter, caseId]);

  const filteredPopulationScores = useMemo(() => {
    const query = populationFilter.trim().toLowerCase();
    if (!query) {
      return populationScores;
    }
    return populationScores.filter((entry) => entry.case_id?.toLowerCase().includes(query));
  }, [populationScores, populationFilter]);

  const runScoreByPatient = useMemo(() => {
    const map = new Map();
    availableRuns.forEach((run) => {
      if (!run.case_id || !run.plan_score?.plan_score) {
        return;
      }
      if (!map.has(run.case_id)) {
        map.set(run.case_id, run.plan_score);
      }
    });
    return map;
  }, [availableRuns]);

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
    setPlanScore(null);
    setPlanScoreStatus("idle");
    setCtMeta(null);
    setCtSliceIndex(0);
    setStructures([]);
    setStructureOverlay(false);
    setStructureSelection("all");
    setDoseOverlay(false);
    setDoseOpacity(0.5);
    setDoseMin(0);
    setDoseMax(null);
    setDoseMeta(null);
    setDoseStatus("idle");
    setRtPlanStatus("idle");
    setRtPlanFile("");
    setRtDoseFile("");
    setCtDicomStatus("idle");
    setCtDicomDir("");
    setRtStructStatus("idle");
    setRtStructPath("");
    resultsLoaded.current = false;
    setIsRunning(true);
    try {
      const payload = {
        case_id: caseId,
        protocol,
        adapter: "example",
        fast: preset === "fast",
        super_fast: preset === "super_fast",
        optimizer,
        compress_mode: compressMode,
        threshold_perc: thresholdPerc,
        rank,
      };
      if (tag.trim()) {
        payload.tag = tag.trim();
      }
      if (optimizer === "compressrtp") {
        const parsedCount = Number.parseInt(String(beamCount).trim(), 10);
        if (Number.isFinite(parsedCount) && parsedCount > 0) {
          const capped = Math.min(parsedCount, 37);
          payload.beam_ids = Array.from({ length: capped }, (_, idx) => idx);
        }
        payload.use_gpu = useGpu;
        if (compressStep && compressStep !== "all") {
          payload.step = compressStep;
        }
      }
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
    setPlanScore(null);
    setPlanScoreStatus("idle");
    setCtMeta(null);
    setCtSliceIndex(0);
    setStructures([]);
    setStructureOverlay(false);
    setStructureSelection("all");
    setDoseOverlay(false);
    setDoseOpacity(0.5);
    setDoseMin(0);
    setDoseMax(null);
    setDoseMeta(null);
    setDoseStatus("idle");
    setRtPlanStatus("idle");
    setRtPlanFile("");
    setRtDoseFile("");
    setCtDicomStatus("idle");
    setCtDicomDir("");
    setRtStructStatus("idle");
    setRtStructPath("");
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

  async function createCtDicom() {
    if (!runId) {
      return;
    }
    setCtDicomStatus("creating");
    setError(null);
    try {
      const response = await fetch(`${apiBase}/runs/${runId}/ct-dicom`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`CT DICOM export failed: ${response.status}`);
      }
      const payload = await response.json();
      if (payload.status === "created" || payload.status === "exists") {
        setCtDicomStatus("ready");
        setCtDicomDir(payload.ct_dir || "");
      } else {
        setCtDicomStatus("idle");
      }
    } catch (err) {
      setCtDicomStatus("error");
      setError(err.message || "Failed to export CT DICOM.");
    }
  }

  async function createRtStruct() {
    if (!runId) {
      return;
    }
    setRtStructStatus("creating");
    setError(null);
    try {
      const response = await fetch(`${apiBase}/runs/${runId}/rtstruct`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`RTSTRUCT export failed: ${response.status}`);
      }
      const payload = await response.json();
      if (payload.status === "created" || payload.status === "exists") {
        setRtStructStatus("ready");
        setRtStructPath(payload.artifact || "");
        if (payload.ct_dir) {
          setCtDicomDir(payload.ct_dir);
        }
      } else {
        setRtStructStatus("idle");
      }
    } catch (err) {
      setRtStructStatus("error");
      setError(err.message || "Failed to export RTSTRUCT.");
    }
  }

  async function createRtPlan() {
    if (!runId) {
      return;
    }
    setRtPlanStatus("creating");
    setError(null);
    try {
      const response = await fetch(`${apiBase}/runs/${runId}/rtplan`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`RT Plan export failed: ${response.status}`);
      }
      const payload = await response.json();
      if (payload.status === "created" || payload.status === "exists") {
        setRtPlanStatus("ready");
        setRtPlanFile(payload.artifact || "");
        if (payload.dose_artifact) {
          setRtDoseFile(payload.dose_artifact);
        }
      } else {
        setRtPlanStatus("idle");
      }
    } catch (err) {
      setRtPlanStatus("error");
      setError(err.message || "Failed to export RT Plan.");
    }
  }

  async function loadComparison() {
    if (!compareRunA || !compareRunB) {
      return;
    }
    setCompareLoading(true);
    setCompareError(null);
    try {
      const [dvhARes, dvhBRes, metricsARes, metricsBRes] = await Promise.all([
        fetch(`${apiBase}/runs/${compareRunA}/artifacts/dvh.json`),
        fetch(`${apiBase}/runs/${compareRunB}/artifacts/dvh.json`),
        fetch(`${apiBase}/runs/${compareRunA}/artifacts/metrics.json`),
        fetch(`${apiBase}/runs/${compareRunB}/artifacts/metrics.json`),
      ]);
      if (!dvhARes.ok || !dvhBRes.ok) {
        throw new Error("Failed to load DVH data for comparison.");
      }
      const [dvhA, dvhB] = await Promise.all([dvhARes.json(), dvhBRes.json()]);
      setCompareDvhA(dvhA);
      setCompareDvhB(dvhB);
      const metricsA = metricsARes.ok ? await metricsARes.json() : {};
      const metricsB = metricsBRes.ok ? await metricsBRes.json() : {};
      setCompareMetricsA(metricsA.records || metricsA.metrics || []);
      setCompareMetricsB(metricsB.records || metricsB.metrics || []);
    } catch (err) {
      setCompareError(err.message || "Comparison failed.");
    } finally {
      setCompareLoading(false);
    }
  }

  function clearComparison() {
    setCompareRunA("");
    setCompareRunB("");
    setCompareDvhA(null);
    setCompareDvhB(null);
    setCompareMetricsA([]);
    setCompareMetricsB([]);
    setCompareError(null);
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
      setStructures([]);
      return;
    }
    fetch(`${apiBase}/runs/${runId}/structures`)
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => setStructures(payload?.structures || []))
      .catch(() => setStructures([]));
  }, [apiBase, runId]);

  useEffect(() => {
    if (structureSelection === "all") {
      return;
    }
    if (structures.length && !structures.includes(structureSelection)) {
      setStructureSelection("all");
    }
  }, [structures, structureSelection]);

  useEffect(() => {
    if (!runId) {
      return undefined;
    }
    let active = true;
    let intervalId = null;
    const stopPolling = () => {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    };
    const poll = async () => {
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
          stopPolling();
          return;
        }
        if (payload.status?.state === "completed") {
          if (!resultsLoaded.current) {
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
          setIsRunning(false);
          stopPolling();
        }
      } catch (err) {
        if (active) {
          setError(err.message || "Failed to fetch run status.");
          setIsRunning(false);
          stopPolling();
        }
      }
    };
    intervalId = setInterval(poll, 2000);
    poll();

    return () => {
      active = false;
      stopPolling();
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
    const planName = artifacts.find((name) => name.endsWith(".dcm") && name.startsWith("rt_plan_"));
    const doseName = artifacts.find((name) => name.endsWith(".dcm") && name.startsWith("rt_dose_"));
    if (planName) {
      setRtPlanStatus("ready");
      setRtPlanFile(planName);
      if (doseName) {
        setRtDoseFile(doseName);
      }
      return;
    }
    if (rtPlanStatus === "ready") {
      setRtPlanStatus("idle");
      setRtPlanFile("");
      setRtDoseFile("");
    }
  }, [artifacts, rtPlanStatus]);

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

  useEffect(() => {
    if (!runId) {
      setPlanScore(null);
      setPlanScoreStatus("idle");
      return;
    }
    if (status?.state !== "completed" && !artifacts.includes("plan_score.json")) {
      return;
    }
    setPlanScoreStatus("loading");
    fetch(`${apiBase}/runs/${runId}/plan-score`)
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => {
        if (payload) {
          setPlanScore(payload);
          setPlanScoreStatus("ready");
        } else {
          setPlanScore(null);
          setPlanScoreStatus("idle");
        }
      })
      .catch(() => {
        setPlanScore(null);
        setPlanScoreStatus("error");
      });
  }, [apiBase, runId, status?.state, artifacts]);

  const structureNames = useMemo(() => {
    const names = new Set();
    [dvhData, compareDvhA, compareDvhB].forEach((dataset) => {
      if (!dataset) {
        return;
      }
      Object.keys(dataset).forEach((name) => names.add(name));
    });
    return Array.from(names).sort();
  }, [dvhData, compareDvhA, compareDvhB]);

  const dvhColorMap = useMemo(() => {
    const map = new Map();
    structureNames.forEach((name, idx) => {
      map.set(name, DVH_COLORS[idx % DVH_COLORS.length]);
    });
    return map;
  }, [structureNames]);

  const baseSeries = useMemo(
    () => buildDvhSeries(dvhData, "Current", dvhColorMap, false),
    [dvhData, dvhColorMap]
  );

  const compareSeriesA = useMemo(
    () => buildDvhSeries(compareDvhA, "Run A", dvhColorMap, false),
    [compareDvhA, dvhColorMap]
  );

  const compareSeriesB = useMemo(
    () => buildDvhSeries(compareDvhB, "Run B", dvhColorMap, true),
    [compareDvhB, dvhColorMap]
  );

  const combinedSeries = useMemo(
    () => [...baseSeries, ...compareSeriesA, ...compareSeriesB],
    [baseSeries, compareSeriesA, compareSeriesB]
  );

  const dvhSeries = useMemo(() => combinedSeries, [combinedSeries]);

  const deltaMetrics = useMemo(() => {
    if (!compareMetricsA.length || !compareMetricsB.length) {
      return [];
    }
    return buildDeltaRows(compareMetricsA, compareMetricsB);
  }, [compareMetricsA, compareMetricsB]);

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
    const stageMap = new Map();
    events.forEach((event) => {
      if (!event.stage) {
        return;
      }
      const info = stageMap.get(event.stage) || { done: false, active: false, duration_sec: null };
      const state = event.data?.state;
      if (state === "start") {
        info.active = true;
      }
      if (state === "done") {
        info.done = true;
        info.active = false;
        if (Number.isFinite(event.data?.duration_sec)) {
          info.duration_sec = event.data.duration_sec;
        }
      }
      stageMap.set(event.stage, info);
    });
    const activeStage = status?.stage;
    return STAGES.map((stage) => {
      const info = stageMap.get(stage.id) || {};
      return {
        ...stage,
        done: Boolean(info.done),
        active: Boolean(info.active || (activeStage && activeStage === stage.id)),
        duration_sec: info.duration_sec ?? null,
      };
    });
  }, [events, status]);

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
  const activeStageLabel = status?.stage || lastEvent?.stage || "--";
  const elapsedLabel = formatDuration(status?.elapsed_sec);
  const rssLabel = status?.rss_mb ? `${Number(status.rss_mb).toFixed(1)} MB` : "--";
  const sliceCount = ctMeta?.slice_count || 0;
  const sliceMax = sliceCount > 0 ? sliceCount - 1 : 0;
  const doseReady = doseStatus === "ready";
  const ctImageUrl = useMemo(() => {
    if (!runId || !ctMeta) {
      return "";
    }
    return `${apiBase}/runs/${runId}/ct/slice?index=${ctSliceIndex}&window=${ctWindow}&level=${ctLevel}`;
  }, [apiBase, runId, ctMeta, ctSliceIndex, ctWindow, ctLevel]);
  const structureImageUrl = useMemo(() => {
    if (!runId || !structureOverlay) {
      return "";
    }
    const namesParam =
      structureSelection && structureSelection !== "all"
        ? `&names=${encodeURIComponent(structureSelection)}`
        : "";
    return `${apiBase}/runs/${runId}/structures/slice?index=${ctSliceIndex}${namesParam}`;
  }, [apiBase, runId, structureOverlay, structureSelection, ctSliceIndex]);
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
    const now = Date.now();
    if (now - wheelThrottle.current < 30) {
      return;
    }
    wheelThrottle.current = now;
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
            <input
              id="case-filter"
              type="text"
              placeholder="Search patients"
              value={caseFilter}
              onChange={(event) => setCaseFilter(event.target.value)}
            />
            <select
              id="case"
              value={caseId}
              onChange={(event) => setCaseId(event.target.value)}
              disabled={!caseOptions.length}
            >
              {filteredCaseOptions.length ? (
                filteredCaseOptions.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))
              ) : (
                <option value="">No patients loaded</option>
              )}
            </select>
          </div>
          <span className={`status-pill ${status?.state === "error" ? "error" : ""}`}>
            {status?.state || "idle"}
          </span>
          <div className="status-meta">
            <span>Stage: {activeStageLabel}</span>
            <span>Elapsed: {elapsedLabel}</span>
            <span>RSS: {rssLabel}</span>
          </div>
          <button className="btn btn-ghost" onClick={() => loadRun(selectedRunId)}>
            Load Run
          </button>
          <button className="btn btn-primary" onClick={startRun} disabled={isRunning || !caseId}>
            {isRunning ? "Running" : "Run Example"}
          </button>
        </div>
      </header>

      <div className="tabbar">
        <button
          type="button"
          className={`tab ${activeTab === "workbench" ? "active" : ""}`}
          onClick={() => setActiveTab("workbench")}
        >
          Workbench
        </button>
        <button
          type="button"
          className={`tab ${activeTab === "population" ? "active" : ""}`}
          onClick={() => setActiveTab("population")}
        >
          Plan Score Population
        </button>
      </div>

      <main className="dashboard">
        {activeTab === "workbench" ? (
          <>
            <section className="panel">
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">CT Viewer (Axial)</div>
                <div className="card-subtitle">Slice navigation + window/level</div>
              </div>
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
                  {structureOverlay && structureImageUrl ? (
                    <img
                      className="ct-structures"
                      src={structureImageUrl}
                      alt="Structure overlay"
                      draggable={false}
                    />
                  ) : null}
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
                      <button
                        id="dose-overlay"
                        type="button"
                        className={`btn btn-toggle ${doseOverlay ? "active" : ""}`}
                        onClick={() => setDoseOverlay((prev) => !prev)}
                        disabled={!artifacts.includes("dose_3d.npy")}
                      >
                        {doseOverlay ? "Dose Overlay On" : "Dose Overlay Off"}
                      </button>
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
                      <label htmlFor="structure-overlay">Structure Overlay</label>
                      <button
                        id="structure-overlay"
                        type="button"
                        className={`btn btn-toggle ${structureOverlay ? "active" : ""}`}
                        onClick={() => setStructureOverlay((prev) => !prev)}
                        disabled={!structures.length}
                      >
                        {structureOverlay ? "Structure Overlay On" : "Structure Overlay Off"}
                      </button>
                    </div>
                    <div className="ct-control">
                      <label htmlFor="structure-select">Structure</label>
                      <select
                        id="structure-select"
                        value={structureSelection}
                        onChange={(event) => setStructureSelection(event.target.value)}
                        disabled={!structureOverlay || !structures.length}
                      >
                        <option value="all">All</option>
                        {structures.map((name) => (
                          <option key={name} value={name}>
                            {name}
                          </option>
                        ))}
                      </select>
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
            {rtPlanStatus === "ready" && rtPlanFile ? (
              <div className="card-subtitle" style={{ marginTop: "8px" }}>
                RT Plan saved:{" "}
                <a href={`${apiBase}/runs/${runId}/artifacts/${rtPlanFile}`} target="_blank" rel="noreferrer">
                  {rtPlanFile}
                </a>
              </div>
            ) : null}
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">DICOM Exports</div>
                <div className="card-subtitle">CT/RTSTRUCT + RTPLAN/RTDOSE</div>
              </div>
              <div className="header-actions">
                <button
                  className="btn"
                  onClick={createCtDicom}
                  disabled={!runId || ctDicomStatus === "creating"}
                >
                  {ctDicomStatus === "ready"
                    ? "CT DICOM Ready"
                    : ctDicomStatus === "creating"
                      ? "Exporting CT..."
                      : "Export CT DICOM"}
                </button>
                <button
                  className="btn"
                  onClick={createRtStruct}
                  disabled={!runId || rtStructStatus === "creating"}
                >
                  {rtStructStatus === "ready"
                    ? "RTSTRUCT Ready"
                    : rtStructStatus === "creating"
                      ? "Exporting RTSTRUCT..."
                      : "Export RTSTRUCT"}
                </button>
              </div>
            </div>
            <div className="export-grid">
              <div>
                <div className="export-label">3D Dose Grid</div>
                <div className="export-value">
                  {doseReady ? "dose_3d.npy" : "Not generated"}
                </div>
                <button
                  className="btn btn-ghost"
                  onClick={createDose3d}
                  disabled={!runId || doseReady || doseStatus === "creating"}
                >
                  {doseReady
                    ? "Dose Ready"
                    : doseStatus === "creating"
                      ? "Creating Dose..."
                      : "Create 3D Dose"}
                </button>
              </div>
              <div>
                <div className="export-label">RT Plan + RT Dose</div>
                <div className="export-value">
                  {rtPlanFile ? `${rtPlanFile}${rtDoseFile ? ` + ${rtDoseFile}` : ""}` : "Not generated"}
                </div>
                <button
                  className="btn btn-ghost"
                  onClick={createRtPlan}
                  disabled={!runId || rtPlanStatus === "creating"}
                >
                  {rtPlanStatus === "ready"
                    ? "RT Plan Ready"
                    : rtPlanStatus === "creating"
                      ? "Generating RT Plan..."
                      : "Generate RT Plan"}
                </button>
              </div>
            </div>
            {ctDicomDir ? (
              <div className="card-subtitle" style={{ marginTop: "8px" }}>
                CT DICOM: {ctDicomDir}
              </div>
            ) : null}
            {rtStructPath ? (
              <div className="card-subtitle">RTSTRUCT: {rtStructPath}</div>
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
                      <span>
                        {run.tag ? `${run.tag} · ` : ""}
                        {run.case_id || run.status?.case_id || "unknown"}
                      </span>
                      <span>{run.run_type || "echo-vmat"} · {run.status?.state || "unknown"}</span>
                      <span>
                        {Number.isFinite(run.plan_score?.plan_score)
                          ? `Score ${run.plan_score.plan_score.toFixed(1)}`
                          : ""}
                      </span>
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
              <label htmlFor="optimizer">Optimizer</label>
              <select
                id="optimizer"
                value={optimizer}
                onChange={(event) => setOptimizer(event.target.value)}
              >
                {OPTIMIZERS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
              {optimizer === "compressrtp" ? (
                <>
                  <label htmlFor="compress-mode">Compression</label>
                  <select
                    id="compress-mode"
                    value={compressMode}
                    onChange={(event) => setCompressMode(event.target.value)}
                  >
                    {COMPRESS_MODES.map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <label htmlFor="compress-step">Pipeline Step</label>
                  <select
                    id="compress-step"
                    value={compressStep}
                    onChange={(event) => setCompressStep(event.target.value)}
                  >
                    {COMPRESS_STEPS.map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <div style={{ color: "#94a3b8", fontSize: "12px" }}>
                    Use DDC only for fast validation runs.
                  </div>
                  <label htmlFor="threshold-perc">Threshold %</label>
                  <input
                    id="threshold-perc"
                    type="number"
                    min="1"
                    max="50"
                    step="1"
                    value={thresholdPerc}
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      if (Number.isFinite(value)) {
                        setThresholdPerc(value);
                      }
                    }}
                  />
                  <label htmlFor="beam-count">Beam count (test)</label>
                  <input
                    id="beam-count"
                    type="number"
                    min="1"
                    max="37"
                    placeholder="Leave blank for PlannerBeams.json"
                    value={beamCount}
                    onChange={(event) => setBeamCount(event.target.value)}
                  />
                  <div style={{ color: "#94a3b8", fontSize: "12px" }}>
                    Start with 3 for quick DDC checks; use 7 for closer planner behavior.
                  </div>
                  <label>GPU (optional)</label>
                  <button
                    type="button"
                    className={`btn btn-toggle ${useGpu ? "active" : ""}`}
                    onClick={() => setUseGpu((prev) => !prev)}
                  >
                    {useGpu ? "GPU enabled" : "GPU off"}
                  </button>
                  <div style={{ color: "#94a3b8", fontSize: "12px" }}>
                    Uses GPU for supported matrix ops only; solver remains CPU.
                  </div>
                  {compressMode === "sparse-plus-low-rank" ? (
                    <>
                      <label htmlFor="rank">Low-rank k</label>
                      <input
                        id="rank"
                        type="number"
                        min="1"
                        max="20"
                        step="1"
                        value={rank}
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          if (Number.isFinite(value)) {
                            setRank(value);
                          }
                        }}
                      />
                    </>
                  ) : null}
                </>
              ) : null}
              <label htmlFor="protocol">Protocol</label>
              <input
                id="protocol"
                value={protocol}
                onChange={(event) => setProtocol(event.target.value)}
              />
              <label htmlFor="run-tag">Tag</label>
              <input
                id="run-tag"
                placeholder="Optional label"
                value={tag}
                onChange={(event) => setTag(event.target.value)}
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
                    {run.run_id} ({run.run_type || "echo-vmat"} · {run.status?.state || "unknown"}
                    {run.tag ? ` · ${run.tag}` : ""}
                    {Number.isFinite(run.plan_score?.plan_score)
                      ? ` · Score ${run.plan_score.plan_score.toFixed(1)}`
                      : ""})
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
              {stageStatus.map((stage) => {
                let statusText = "pending";
                if (stage.done) {
                  statusText = stage.duration_sec ? formatDuration(stage.duration_sec) : "done";
                } else if (stage.active) {
                  statusText = "active";
                }
                return (
                  <li
                    key={stage.id}
                    className={`progress-item ${stage.done ? "done" : ""} ${stage.active ? "active" : ""}`}
                  >
                    <span>{stage.label}</span>
                    <span>{statusText}</span>
                  </li>
                );
              })}
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
            <div className="dvh-toolbar">
              <div className="dvh-legend">
                {structureNames.slice(0, 8).map((name) => (
                  <span key={name} className="dvh-legend-item">
                    <span
                      className="dvh-swatch"
                      style={{ background: dvhColorMap.get(name) || "#5bbcff" }}
                    />
                    {name}
                  </span>
                ))}
                {structureNames.length > 8 ? (
                  <span className="dvh-legend-more">+{structureNames.length - 8} more</span>
                ) : null}
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
                <div className="card-title">Plan Score (Population-Based)</div>
                <div className="card-subtitle">Percentile scoring across 99 lung plans</div>
              </div>
              <span className={`status-pill ${planScoreStatus === "error" ? "error" : ""}`}>
                {planScoreStatus}
              </span>
            </div>
            {planScore ? (
              <div className="plan-score-grid">
                <div className="plan-score-daisy">
                  <DaisyPlot score={planScore} objectives={planScore.objectives || []} />
                </div>
                <div className="plan-score-table">
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Objective</th>
                          <th>Value</th>
                          <th>Percentile</th>
                          <th>Target</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(planScore.objectives || []).map((obj) => (
                          <tr key={obj.id}>
                            <td>{obj.label}</td>
                            <td>
                              {obj.value !== null && obj.value !== undefined
                                ? `${obj.value.toFixed(2)} ${obj.unit}`
                                : "--"}
                            </td>
                            <td className="metric-percent">
                              {obj.percentile !== null && obj.percentile !== undefined
                                ? formatPercent(obj.percentile)
                                : "--"}
                            </td>
                            <td>
                              {obj.target_value !== null && obj.target_value !== undefined
                                ? `${obj.target_value} ${obj.unit}`
                                : "--"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            ) : (
              <div className="placeholder">
                {planScoreStatus === "loading"
                  ? "Scoring plan against population..."
                  : "Plan score not available yet."}
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Run Comparison</div>
                <div className="card-subtitle">Overlay DVHs + metric deltas</div>
              </div>
              <div className="comparison-actions">
                <button className="btn" onClick={loadComparison} disabled={compareLoading}>
                  {compareLoading ? "Loading..." : "Load Comparison"}
                </button>
                <button className="btn btn-ghost" onClick={clearComparison} disabled={!compareRunA && !compareRunB}>
                  Clear
                </button>
              </div>
            </div>
            <div className="comparison-grid">
              <div className="comparison-control">
                <label htmlFor="compare-run-a">Run A</label>
                <select
                  id="compare-run-a"
                  value={compareRunA}
                  onChange={(event) => setCompareRunA(event.target.value)}
                >
                  <option value="">Select run</option>
                  {availableRuns.map((run) => (
                    <option key={`a-${run.run_id}`} value={run.run_id}>
                      {run.run_id} ({run.run_type || "echo-vmat"}{run.tag ? ` · ${run.tag}` : ""})
                    </option>
                  ))}
                </select>
              </div>
              <div className="comparison-control">
                <label htmlFor="compare-run-b">Run B</label>
                <select
                  id="compare-run-b"
                  value={compareRunB}
                  onChange={(event) => setCompareRunB(event.target.value)}
                >
                  <option value="">Select run</option>
                  {availableRuns.map((run) => (
                    <option key={`b-${run.run_id}`} value={run.run_id}>
                      {run.run_id} ({run.run_type || "echo-vmat"}{run.tag ? ` · ${run.tag}` : ""})
                    </option>
                  ))}
                </select>
              </div>
            </div>
            {compareError ? <div className="placeholder">{compareError}</div> : null}
            {deltaMetrics.length ? (
              <div className="table-wrap" style={{ marginTop: "10px" }}>
                <table>
                  <thead>
                    <tr>
                      <th>Structure</th>
                      <th>Constraint</th>
                      <th>Run A</th>
                      <th>Run B</th>
                      <th>Delta (A-B)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {deltaMetrics.map((row, idx) => (
                      <tr key={`delta-${idx}`}>
                        <td>{row.structure}</td>
                        <td>{row.constraint}</td>
                        <td>{row.planA !== null ? row.planA.toFixed(2) : "--"}</td>
                        <td>{row.planB !== null ? row.planB.toFixed(2) : "--"}</td>
                        <td className={row.delta > 0 ? "delta-up" : row.delta < 0 ? "delta-down" : ""}>
                          {row.delta !== null ? row.delta.toFixed(2) : "--"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="placeholder" style={{ marginTop: "10px" }}>
                Load two runs to compare metrics.
              </div>
            )}
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

          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Resource Summary</div>
                <div className="card-subtitle">Peak memory per stage</div>
              </div>
            </div>
            <div className="kpi-grid">
              {RESOURCE_LABELS.map((item) => (
                <div className="kpi" key={item.key}>
                  <div className="label">{item.label}</div>
                  <div className="value">{formatNumber(timing?.[item.key])}</div>
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
          </>
        ) : null}

        {activeTab === "population" ? (
          <section className="panel">
            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Population Plan Score Variation</div>
                  <div className="card-subtitle">Reference distribution across {populationScores.length || "--"} plans</div>
                </div>
                <span className={`status-pill ${populationStatus === "error" ? "error" : ""}`}>
                  {populationStatus}
                </span>
              </div>
              {populationScores.length ? (
                <div className="population-grid">
                  <div>
                    <PopulationHistogram
                      values={populationScores.map((entry) => entry.plan_score ?? 0)}
                      marker={planScore?.plan_score ?? null}
                    />
                    <div className="population-legend">
                      <span>Blue = reference plans</span>
                      <span>Orange = selected run</span>
                    </div>
                  </div>
                  <div className="population-stats">
                    <div className="stat">
                      <span>Mean</span>
                      <strong>
                        {Number.isFinite(populationStats?.mean) ? populationStats.mean.toFixed(2) : "--"}
                      </strong>
                    </div>
                    <div className="stat">
                      <span>Median</span>
                      <strong>
                        {Number.isFinite(populationStats?.median) ? populationStats.median.toFixed(2) : "--"}
                      </strong>
                    </div>
                    <div className="stat">
                      <span>P10</span>
                      <strong>
                        {Number.isFinite(populationStats?.p10) ? populationStats.p10.toFixed(2) : "--"}
                      </strong>
                    </div>
                    <div className="stat">
                      <span>P90</span>
                      <strong>
                        {Number.isFinite(populationStats?.p90) ? populationStats.p90.toFixed(2) : "--"}
                      </strong>
                    </div>
                    <div className="stat">
                      <span>Min</span>
                      <strong>
                        {Number.isFinite(populationStats?.min) ? populationStats.min.toFixed(2) : "--"}
                      </strong>
                    </div>
                    <div className="stat">
                      <span>Max</span>
                      <strong>
                        {Number.isFinite(populationStats?.max) ? populationStats.max.toFixed(2) : "--"}
                      </strong>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="placeholder">
                  {populationStatus === "loading"
                    ? "Loading population scores..."
                    : "Population scores not available."}
                </div>
              )}
            </div>

            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">Patient Plan Scores</div>
                  <div className="card-subtitle">Reference RTDOSE plan score for each patient</div>
                </div>
              </div>
              <div className="population-toolbar">
                <input
                  type="text"
                  placeholder="Filter by patient ID"
                  value={populationFilter}
                  onChange={(event) => setPopulationFilter(event.target.value)}
                />
                <div className="population-count">
                  Showing {filteredPopulationScores.length} / {populationScores.length}
                </div>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Patient</th>
                      <th>Reference Score</th>
                      <th>Percentile</th>
                      <th>Latest Run Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPopulationScores.length ? (
                      filteredPopulationScores.map((entry) => {
                        const runScore = runScoreByPatient.get(entry.case_id);
                        return (
                          <tr key={entry.case_id}>
                            <td>{entry.case_id}</td>
                            <td>{Number.isFinite(entry.plan_score) ? entry.plan_score.toFixed(2) : "--"}</td>
                            <td>{entry.percentile !== null ? formatPercent(entry.percentile) : "--"}</td>
                            <td>
                              {Number.isFinite(runScore?.plan_score) ? runScore.plan_score.toFixed(2) : "--"}
                            </td>
                          </tr>
                        );
                      })
                    ) : (
                      <tr>
                        <td colSpan={4} className="placeholder">
                          No patient scores found.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        ) : null}
      </main>
    </div>
  );
}
