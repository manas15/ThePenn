/**
 * chart.js — Canvas-based live accelerometer chart.
 * Draws DX (red), DY (green), DZ (blue) lines over a rolling time window.
 * Auto-scales the Y axis to fit the visible data with padding.
 */

class AccelChart {
    constructor(canvas, { windowSec = 3, autoScale = true, yMin = -0.1, yMax = 0.1 } = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.windowSec = windowSec;
        this.autoScale = autoScale;
        this.yMin = yMin;
        this.yMax = yMax;
        this.data = []; // [{x, y, z, t}]
        this.maxPoints = 200;
        this.recording = false;

        this._resize();
        this._resizeObserver = new ResizeObserver(() => this._resize());
        this._resizeObserver.observe(canvas);
    }

    _resize() {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.w = rect.width;
        this.h = rect.height;
        this.draw();
    }

    push(sample) {
        this.data.push(sample);
        if (this.data.length > this.maxPoints) {
            this.data.shift();
        }
    }

    setData(samples) {
        this.data = samples.slice(-this.maxPoints);
    }

    clear() {
        this.data = [];
    }

    draw() {
        const { ctx, w, h, data, windowSec } = this;
        ctx.clearRect(0, 0, w, h);

        if (data.length < 2) return;

        // Recording tint
        if (this.recording) {
            ctx.fillStyle = "rgba(63, 185, 80, 0.1)";
            ctx.fillRect(0, 0, w, h);
        }

        // Time range
        const tEnd = data[data.length - 1].t;
        const tStart = tEnd - windowSec;

        // Auto-scale: find min/max of visible data
        let yMin = this.yMin;
        let yMax = this.yMax;
        if (this.autoScale) {
            let lo = Infinity, hi = -Infinity;
            for (const pt of data) {
                if (pt.t < tStart) continue;
                for (const k of ["x", "y", "z"]) {
                    if (pt[k] < lo) lo = pt[k];
                    if (pt[k] > hi) hi = pt[k];
                }
            }
            if (lo !== Infinity) {
                const range = hi - lo;
                const pad = Math.max(range * 0.3, 0.005); // at least 0.005g padding
                yMin = lo - pad;
                yMax = hi + pad;
                // Keep it symmetric around zero
                const absMax = Math.max(Math.abs(yMin), Math.abs(yMax));
                yMin = -absMax;
                yMax = absMax;
            }
        }

        // Grid lines
        ctx.strokeStyle = "#21262d";
        ctx.lineWidth = 1;
        const gridLines = 5;
        for (let i = 0; i <= gridLines; i++) {
            const gy = (i / gridLines) * h;
            ctx.beginPath();
            ctx.moveTo(0, gy);
            ctx.lineTo(w, gy);
            ctx.stroke();
        }

        // Zero line (brighter)
        const zeroY = h * (yMax / (yMax - yMin));
        ctx.strokeStyle = "#484f58";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, zeroY);
        ctx.lineTo(w, zeroY);
        ctx.stroke();

        // Y-axis scale label
        ctx.fillStyle = "#484f58";
        ctx.font = "10px monospace";
        ctx.fillText(`±${yMax.toFixed(3)}g`, 4, 12);

        const toX = (t) => ((t - tStart) / windowSec) * w;
        const toY = (v) => h - ((v - yMin) / (yMax - yMin)) * h;

        // Draw each axis
        const axes = [
            { key: "x", color: "#f87171" },
            { key: "y", color: "#4ade80" },
            { key: "z", color: "#60a5fa" },
        ];

        for (const axis of axes) {
            ctx.strokeStyle = axis.color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            let started = false;
            for (const pt of data) {
                if (pt.t < tStart) continue;
                const px = toX(pt.t);
                const py = toY(pt[axis.key]);
                if (!started) {
                    ctx.moveTo(px, py);
                    started = true;
                } else {
                    ctx.lineTo(px, py);
                }
            }
            ctx.stroke();
        }
    }
}
