import { useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
} from "recharts";

const RAW_CSV = `timestamp,elapsed_s,x_g,y_g,z_g
2026-04-04T16:33:02.409,0.345,-69.000,0.123,0.969
2026-04-04T16:33:02.450,0.386,-0.173,0.120,71.000
2026-04-04T16:33:11.108,9.044,1.000,0.791,-18.000
2026-04-04T16:33:14.712,12.648,40.000,0.000,0.143
2026-04-04T16:33:17.522,15.458,0.321,0.000,4.000
2026-04-04T16:33:20.221,18.157,-0.192,8.000,0.000
2026-04-04T16:33:24.181,22.117,0.293,0.900,3.000
2026-04-04T16:33:27.663,25.599,44.000,0.000,74.000
2026-04-04T16:33:28.515,26.451,0.305,0.019,0.400
2026-04-04T16:33:28.871,26.807,0.635,0.609,0.000
2026-04-04T16:33:46.413,44.349,0.239,0.000,37.000
2026-04-04T16:33:49.148,47.085,0.364,923.000,0.700
2026-04-04T16:33:55.837,53.773,0.459,5.000,0.000
2026-04-04T16:33:55.853,53.789,0.558,0.000,0.251
2026-04-04T16:34:02.816,60.752,0.366,0.000,0.329
2026-04-04T16:34:12.428,70.364,0.372,0.000,0.000
2026-04-04T16:34:16.311,74.247,198.000,0.976,0.000
2026-04-04T16:34:38.362,96.298,0.406,0.600,0.000
2026-04-04T16:34:44.210,102.147,0.532,0.707,0.000
2026-04-04T16:34:46.168,104.104,0.575,3.000,0.000
2026-04-04T16:34:46.639,104.575,5042978.000,0.689,0.438
2026-04-04T16:34:51.660,109.597,3.000,0.646,5.000
2026-04-04T16:34:51.722,109.658,0.645,0.659,0.312
2026-04-04T16:35:07.109,125.045,1.000,0.555,3.000
2026-04-04T16:35:08.711,126.647,27.000,0.555,-0.640
2026-04-04T16:35:21.317,139.253,0.501,1.000,-0.682
2026-04-04T16:35:23.107,141.043,0.499,0.000,6.000
2026-04-04T16:35:23.668,141.604,0.478,0.531,-0.000
2026-04-04T16:35:25.208,143.144,0.475,0.000,-0.702
2026-04-04T16:35:26.768,144.705,0.474,0.000,6.000
2026-04-04T16:35:27.620,145.557,0.479,0.527,-0.000
2026-04-04T16:35:32.261,150.197,72.000,0.529,-0.700
2026-04-04T16:35:32.842,150.778,0.468,5.000,-0.000
2026-04-04T16:35:36.713,154.649,76.000,0.527,-0.694
2026-04-04T16:35:39.813,157.749,75.000,0.517,-0.708
2026-04-04T16:35:40.063,157.999,0.473,0.000,7.000
2026-04-04T16:35:41.935,159.871,0.479,0.000,712.000
2026-04-04T16:35:43.159,161.095,6.000,0.526,-0.000
2026-04-04T16:35:46.759,164.695,0.433,5.000,-0.724
2026-04-04T16:35:46.800,164.736,5.000,0.504,-33.000
2026-04-04T16:35:56.204,174.140,0.000,0.311,-0.856
2026-04-04T16:36:16.408,194.344,0.298,0.000,-0.000
2026-04-04T16:36:16.510,194.446,0.309,0.300,-0.000
2026-04-04T16:36:38.958,216.895,283.000,0.320,-0.000
2026-04-04T16:36:48.383,226.319,0.277,0.314,-0.000
2026-04-04T16:36:55.374,233.310,89675.000,0.000,-0.899
2026-04-04T16:37:00.719,238.655,0.280,0.314,-3.000
2026-04-04T16:37:18.363,256.299,0.275,0.000,9.000
2026-04-04T16:37:31.198,269.135,77.000,0.000,-0.000
2026-04-04T16:38:00.822,298.758,0.229,1.000,-0.000
2026-04-04T16:38:01.572,299.508,0.230,0.302,-0.000
2026-04-04T16:38:03.276,301.212,0.000,0.299,-0.232
2026-04-04T16:38:03.796,301.732,0.000,0.900,0.080
2026-04-04T16:38:08.563,306.499,0.228,0.000,-0.928
2026-04-04T16:38:12.806,310.742,0.222,0.297,-0.928
2026-04-04T16:38:17.467,315.403,0.212,0.310,-0.940
2026-04-04T16:38:18.507,316.443,34.000,0.290,9.000
2026-04-04T16:38:21.108,319.044,26.000,0.000,9.000
2026-04-04T16:38:23.209,321.145,46.000,0.270,903.000
2026-04-04T16:38:23.230,321.166,0.213,0.274,-0.000
2026-04-04T16:38:29.262,327.199,0.177,0.000,-0.950
2026-04-04T16:38:32.924,330.860,0.180,0.229,-0.946
2026-04-04T16:38:32.944,330.881,0.183,0.234,-0.949
2026-04-04T16:38:32.965,330.901,0.180,1.000,-0.000
2026-04-04T16:38:33.112,331.049,0.177,0.229,-0.947
2026-04-04T16:38:40.558,338.494,0.181,0.000,-0.000
2026-04-04T16:39:11.829,369.765,0.180,0.000,-0.000
2026-04-04T16:39:33.007,390.943,0.165,0.247,-0.942
2026-04-04T16:39:35.002,392.938,0.000,49.000,-4.000
2026-04-04T16:39:35.211,393.147,0.160,47.000,-8.000
2026-04-04T16:39:55.869,413.805,0.142,48.000,-0.000
2026-04-04T16:40:05.961,423.897,0.143,0.240,-0.951
2026-04-04T16:40:18.506,436.442,35.000,0.241,-0.951
2026-04-04T16:40:18.567,436.503,0.139,36.000,0.000
2026-04-04T16:40:24.330,442.266,0.138,0.247,-0.956
2026-04-04T16:40:29.843,447.779,0.140,0.257,-0.952
2026-04-04T16:40:30.010,447.947,0.138,0.250,-0.000
2026-04-04T16:40:30.092,448.029,0.141,1.000,-0.956
2026-04-04T16:40:40.618,458.554,0.130,48.000,-0.954
2026-04-04T16:40:48.609,466.545,0.146,4.000,-0.963
2026-04-04T16:40:48.859,466.795,0.158,0.000,-0.000
2026-04-04T16:40:49.608,467.544,0.160,0.239,9.000
2026-04-04T16:40:50.792,468.728,0.159,0.400,-0.957
2026-04-04T16:40:54.851,472.787,0.610,0.239,-0.946
2026-04-04T16:40:55.100,473.037,0.162,0.000,-0.000
2026-04-04T16:41:00.404,478.341,0.157,3.000,-0.950
2026-04-04T16:41:01.567,479.504,0.244,-30.182,-60.000
2026-04-04T16:41:08.559,486.495,0.157,0.241,-3.000
2026-04-04T16:41:08.829,486.765,0.000,46.000,-0.952
2026-04-04T16:41:08.850,486.786,0.164,0.234,-2.000
2026-04-04T16:41:08.891,486.827,0.177,-0.984,0.340
2026-04-04T16:41:08.956,486.892,56.000,0.249,-0.987
2026-04-04T16:41:09.058,486.995,0.160,0.241,-8.000
2026-04-04T16:41:10.807,488.743,58.000,0.000,-0.000
2026-04-04T16:41:11.037,488.973,0.153,0.241,-0.948
2026-04-04T16:41:12.908,490.845,0.150,0.000,-0.000
2026-04-04T16:41:13.908,491.844,0.155,0.246,-0.900
2026-04-04T16:41:18.024,495.960,0.155,0.216,47.000
2026-04-04T16:41:18.962,496.898,0.152,0.247,-0.000
2026-04-04T16:41:20.395,498.331,9.000,0.250,53.000
2026-04-04T16:41:24.266,502.202,0.149,0.245,-7.000
2026-04-04T16:41:31.698,509.634,0.148,0.240,-49.000
2026-04-04T16:41:43.330,521.266,0.150,0.248,-0.951
2026-04-04T16:41:43.350,521.286,0.147,0.245,-0.957
2026-04-04T16:41:43.412,521.348,0.149,0.246,-0.949
2026-04-04T16:41:43.432,521.368,0.147,0.250,-0.955
2026-04-04T16:41:43.452,521.389,0.149,0.246,-0.953
2026-04-04T16:41:43.493,521.430,0.153,0.248,-0.950
2026-04-04T16:41:43.641,521.577,147.000,0.251,-0.949
2026-04-04T16:41:43.661,521.598,0.153,0.242,-0.948
2026-04-04T16:41:43.682,521.618,0.152,0.247,-9.000
2026-04-04T16:41:43.829,521.765,0.153,0.245,-0.947
2026-04-04T16:41:43.850,521.786,0.146,0.245,-0.953
2026-04-04T16:41:47.177,525.113,0.149,0.250,-0.952
2026-04-04T16:41:47.180,525.116,0.147,0.252,-0.945
2026-04-04T16:41:47.200,525.136,0.147,0.247,-0.952
2026-04-04T16:41:47.221,525.157,0.147,0.245,-0.953
2026-04-04T16:41:47.241,525.177,0.147,0.251,-0.947
2026-04-04T16:41:47.261,525.198,0.150,0.246,-0.953
2026-04-04T16:41:47.282,525.218,0.152,0.250,-0.948
2026-04-04T16:41:47.311,525.247,0.147,0.250,-0.958
2026-04-04T16:41:54.044,531.980,0.149,0.000,-9.000
2026-04-04T16:42:00.847,538.783,0.150,0.000,9.000
2026-04-04T16:42:06.462,544.399,47.146,0.244,-9.000
2026-04-04T16:42:13.265,551.202,0.148,0.247,-6.000
2026-04-04T16:42:15.387,553.323,0.000,47.000,-4050.000
2026-04-04T16:42:17.386,555.322,0.146,0.245,-0.951
2026-04-04T16:42:17.406,555.342,0.143,0.248,-0.949
2026-04-04T16:42:17.427,555.363,0.149,0.243,-0.955
2026-04-04T16:42:17.447,555.383,0.147,0.245,-0.955
2026-04-04T16:42:17.468,555.404,0.151,0.250,-0.954
2026-04-04T16:42:17.490,555.426,0.149,0.249,-0.951
2026-04-04T16:42:21.322,559.258,0.000,4.000,9.000
2026-04-04T16:42:31.745,569.682,0.113,45.000,-8.000`;

function parseCSV(csv) {
  const lines = csv.trim().split("\n");
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",");
    if (parts.length < 5) continue;
    rows.push({
      timestamp: parts[0],
      elapsed: parseFloat(parts[1]),
      x: parseFloat(parts[2]),
      y: parseFloat(parts[3]),
      z: parseFloat(parts[4]),
    });
  }
  return rows;
}

function filterOutliers(data, threshold) {
  return data.map((row) => ({
    ...row,
    x: Math.abs(row.x) <= threshold ? row.x : null,
    y: Math.abs(row.y) <= threshold ? row.y : null,
    z: Math.abs(row.z) <= threshold ? row.z : null,
  }));
}

export default function AccelPlot() {
  const allData = useMemo(() => parseCSV(RAW_CSV), []);
  const [threshold, setThreshold] = useState(3);
  const [showX, setShowX] = useState(true);
  const [showY, setShowY] = useState(true);
  const [showZ, setShowZ] = useState(true);

  const filtered = useMemo(
    () => filterOutliers(allData, threshold),
    [allData, threshold]
  );

  const totalSamples = allData.length;
  const outlierCount = filtered.filter(
    (r) => r.x === null || r.y === null || r.z === null
  ).length;

  const cleanSamples = filtered.filter(
    (r) => r.x !== null && r.y !== null && r.z !== null
  );

  const stats = useMemo(() => {
    if (cleanSamples.length === 0)
      return { xAvg: 0, yAvg: 0, zAvg: 0, xMin: 0, xMax: 0, yMin: 0, yMax: 0, zMin: 0, zMax: 0 };
    const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const xs = cleanSamples.map((r) => r.x);
    const ys = cleanSamples.map((r) => r.y);
    const zs = cleanSamples.map((r) => r.z);
    return {
      xAvg: avg(xs).toFixed(3),
      yAvg: avg(ys).toFixed(3),
      zAvg: avg(zs).toFixed(3),
      xMin: Math.min(...xs).toFixed(3),
      xMax: Math.max(...xs).toFixed(3),
      yMin: Math.min(...ys).toFixed(3),
      yMax: Math.max(...ys).toFixed(3),
      zMin: Math.min(...zs).toFixed(3),
      zMax: Math.max(...zs).toFixed(3),
    };
  }, [cleanSamples]);

  const duration = allData.length > 0
    ? (allData[allData.length - 1].elapsed - allData[0].elapsed).toFixed(1)
    : 0;

  return (
    <div style={{ background: "#0f1117", color: "#e0e0e0", minHeight: "100vh", padding: "24px", fontFamily: "'Inter', system-ui, -apple-system, sans-serif" }}>
      <div style={{ maxWidth: 960, margin: "0 auto" }}>
        <div style={{ marginBottom: 24 }}>
          <h1 style={{ fontSize: 22, fontWeight: 600, color: "#fff", margin: 0 }}>
            MMA8452Q Accelerometer Data
          </h1>
          <p style={{ fontSize: 13, color: "#888", margin: "4px 0 0" }}>
            accel_log_20260404_163302.csv &mdash; {totalSamples} samples over {duration}s
          </p>
        </div>

        {/* Controls */}
        <div style={{
          display: "flex", alignItems: "center", gap: 24, marginBottom: 20,
          background: "#1a1d27", borderRadius: 10, padding: "12px 18px", flexWrap: "wrap"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 13, color: "#aaa" }}>Outlier threshold (g):</span>
            <input
              type="range" min="1" max="10" step="0.5"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{ width: 120, accentColor: "#6366f1" }}
            />
            <span style={{ fontSize: 13, fontWeight: 600, color: "#6366f1", minWidth: 30 }}>
              {threshold}
            </span>
          </div>
          <div style={{ fontSize: 12, color: "#f87171" }}>
            {outlierCount} sample{outlierCount !== 1 ? "s" : ""} filtered
          </div>
          <div style={{ display: "flex", gap: 12, marginLeft: "auto" }}>
            {[
              { label: "X", color: "#ef4444", active: showX, toggle: () => setShowX(!showX) },
              { label: "Y", color: "#22c55e", active: showY, toggle: () => setShowY(!showY) },
              { label: "Z", color: "#3b82f6", active: showZ, toggle: () => setShowZ(!showZ) },
            ].map((axis) => (
              <button
                key={axis.label}
                onClick={axis.toggle}
                style={{
                  background: axis.active ? axis.color + "22" : "transparent",
                  border: `1.5px solid ${axis.active ? axis.color : "#333"}`,
                  borderRadius: 6, padding: "4px 14px", cursor: "pointer",
                  color: axis.active ? axis.color : "#555",
                  fontSize: 13, fontWeight: 600, transition: "all 0.15s",
                }}
              >
                {axis.label}
              </button>
            ))}
          </div>
        </div>

        {/* Chart */}
        <div style={{
          background: "#1a1d27", borderRadius: 12, padding: "20px 12px 8px",
          marginBottom: 20
        }}>
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={filtered} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3a" />
              <XAxis
                dataKey="elapsed"
                tick={{ fill: "#888", fontSize: 11 }}
                label={{ value: "Time (s)", position: "insideBottom", offset: -2, fill: "#888", fontSize: 12 }}
                tickFormatter={(v) => v.toFixed(0)}
              />
              <YAxis
                tick={{ fill: "#888", fontSize: 11 }}
                label={{ value: "Acceleration (g)", angle: -90, position: "insideLeft", fill: "#888", fontSize: 12 }}
                domain={[-threshold, threshold]}
              />
              <Tooltip
                contentStyle={{
                  background: "#1e2030", border: "1px solid #333",
                  borderRadius: 8, fontSize: 12, color: "#e0e0e0"
                }}
                labelFormatter={(v) => `t = ${v.toFixed(2)}s`}
                formatter={(value, name) =>
                  value !== null ? [value.toFixed(3) + " g", name.toUpperCase()] : ["filtered", name.toUpperCase()]
                }
              />
              <Legend
                wrapperStyle={{ fontSize: 12, color: "#aaa", paddingTop: 8 }}
              />
              {showX && (
                <Line
                  type="monotone" dataKey="x" name="x" stroke="#ef4444"
                  strokeWidth={1.5} dot={false} connectNulls={false}
                  activeDot={{ r: 4, strokeWidth: 0 }}
                />
              )}
              {showY && (
                <Line
                  type="monotone" dataKey="y" name="y" stroke="#22c55e"
                  strokeWidth={1.5} dot={false} connectNulls={false}
                  activeDot={{ r: 4, strokeWidth: 0 }}
                />
              )}
              {showZ && (
                <Line
                  type="monotone" dataKey="z" name="z" stroke="#3b82f6"
                  strokeWidth={1.5} dot={false} connectNulls={false}
                  activeDot={{ r: 4, strokeWidth: 0 }}
                />
              )}
              <Brush
                dataKey="elapsed" height={28} stroke="#6366f1"
                fill="#13151d" tickFormatter={(v) => v.toFixed(0) + "s"}
                travellerWidth={10}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Stats */}
        <div style={{
          display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12,
        }}>
          {[
            { label: "X-axis", color: "#ef4444", avg: stats.xAvg, min: stats.xMin, max: stats.xMax },
            { label: "Y-axis", color: "#22c55e", avg: stats.yAvg, min: stats.yMin, max: stats.yMax },
            { label: "Z-axis", color: "#3b82f6", avg: stats.zAvg, min: stats.zMin, max: stats.zMax },
          ].map((s) => (
            <div
              key={s.label}
              style={{
                background: "#1a1d27", borderRadius: 10, padding: "14px 18px",
                borderLeft: `3px solid ${s.color}`,
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 600, color: s.color, marginBottom: 8 }}>
                {s.label}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                  <span style={{ color: "#888" }}>Mean</span>
                  <span style={{ fontFamily: "monospace", color: "#e0e0e0" }}>{s.avg} g</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                  <span style={{ color: "#888" }}>Min</span>
                  <span style={{ fontFamily: "monospace", color: "#e0e0e0" }}>{s.min} g</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                  <span style={{ color: "#888" }}>Max</span>
                  <span style={{ fontFamily: "monospace", color: "#e0e0e0" }}>{s.max} g</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        <p style={{ fontSize: 11, color: "#555", textAlign: "center", marginTop: 16 }}>
          Readings outside &plusmn;{threshold}g are treated as serial noise and filtered out.
          Adjust the slider to change the threshold.
        </p>
      </div>
    </div>
  );
}
