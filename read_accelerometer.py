"""
read_accelerometer.py

Reads MMA8452Q accelerometer data streamed from an Arduino Uno over serial,
logs it to a timestamped CSV file, and displays a real-time matplotlib plot.

Requirements:
    pip install pyserial matplotlib

Usage:
    python read_accelerometer.py                     # auto-detect port
    python read_accelerometer.py --port COM3         # Windows
    python read_accelerometer.py --port /dev/ttyACM0 # Linux
    python read_accelerometer.py --port /dev/cu.usbmodem14101  # macOS
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime

import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BAUD_RATE = 9600
MAX_POINTS = 200          # number of data points visible on the live plot
PLOT_INTERVAL_MS = 50     # milliseconds between plot refreshes


def find_arduino_port():
    """Auto-detect the first likely Arduino serial port."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        mfr = (p.manufacturer or "").lower()
        if any(keyword in desc for keyword in ["arduino", "ch340", "usb serial", "acm"]):
            return p.device
        if "arduino" in mfr:
            return p.device
    # Fallback: return first available port if only one exists
    if len(ports) == 1:
        return ports[0].device
    return None


def open_serial(port: str) -> serial.Serial:
    """Open the serial connection and wait for the Arduino to reset."""
    ser = serial.Serial(port, BAUD_RATE, timeout=1)
    time.sleep(2)  # Arduino resets on serial open; wait for it
    ser.reset_input_buffer()
    return ser


def parse_line(raw: bytes):
    """
    Parse a single line from the Arduino.
    Expected format:  X_val<TAB>Y_val<TAB>Z_val
    Returns (x, y, z) as floats, or None if the line is not valid data.
    """
    try:
        text = raw.decode("utf-8", errors="replace").strip()
        if not text or "\t" not in text:
            return None
        parts = text.split("\t")
        if len(parts) != 3:
            return None
        return tuple(float(v) for v in parts)
    except (ValueError, UnicodeDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Read MMA8452Q accelerometer data from Arduino Uno")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (e.g. COM3, /dev/ttyACM0). Auto-detected if omitted.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable the live plot (CSV logging only).")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory where the CSV file will be saved.")
    args = parser.parse_args()

    # --- Resolve serial port ---
    port = args.port or find_arduino_port()
    if port is None:
        print("ERROR: No Arduino found. Plug it in or specify --port manually.")
        print("Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}  –  {p.description}")
        sys.exit(1)

    print(f"Connecting to {port} at {BAUD_RATE} baud …")
    ser = open_serial(port)
    print("Connected! Streaming accelerometer data.\n")

    # --- Prepare CSV ---
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(args.output_dir, f"accel_log_{timestamp_str}.csv")
    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "elapsed_s", "x_g", "y_g", "z_g", "dx", "dy", "dz", "l2_norm"])
    print(f"Logging to: {csv_filename}\n")

    # --- Data buffers for plotting ---
    t_data = deque(maxlen=MAX_POINTS)
    x_data = deque(maxlen=MAX_POINTS)
    y_data = deque(maxlen=MAX_POINTS)
    z_data = deque(maxlen=MAX_POINTS)
    # Delta buffers
    dx_data = deque(maxlen=MAX_POINTS)
    dy_data = deque(maxlen=MAX_POINTS)
    dz_data = deque(maxlen=MAX_POINTS)
    # L2 norm buffer
    l2_data = deque(maxlen=MAX_POINTS)
    prev_reading = [None]  # mutable container for previous (x, y, z)
    start_time = time.time()
    sample_count = [0]

    def read_and_store():
        """Read all available lines from serial buffer, store & log them."""
        while ser.in_waiting:
            line = ser.readline()
            parsed = parse_line(line)
            if parsed is None:
                continue
            x, y, z = parsed
            elapsed = time.time() - start_time
            now_str = datetime.now().isoformat(timespec="milliseconds")

            # Compute deltas and L2 norm
            if prev_reading[0] is not None:
                px, py, pz = prev_reading[0]
                dx = x - px
                dy = y - py
                dz = z - pz
                l2 = dx * dx + dy * dy + dz * dz
            else:
                dx, dy, dz, l2 = 0.0, 0.0, 0.0, 0.0
            prev_reading[0] = (x, y, z)

            # Write to CSV
            csv_writer.writerow([now_str, f"{elapsed:.3f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}",
                                 f"{dx:.4f}", f"{dy:.4f}", f"{dz:.4f}", f"{l2:.6f}"])

            # Buffer for plot
            t_data.append(elapsed)
            x_data.append(x)
            y_data.append(y)
            z_data.append(z)
            dx_data.append(dx)
            dy_data.append(dy)
            dz_data.append(dz)
            l2_data.append(l2)

            sample_count[0] += 1
            if sample_count[0] % 100 == 0:
                csv_file.flush()  # periodic flush so data isn't lost

    # ------------------------------------------------------------------
    # Mode A: CSV only (no plot)
    # ------------------------------------------------------------------
    if args.no_plot:
        print("Live plot disabled. Press Ctrl+C to stop.\n")
        prev = None
        try:
            while True:
                line = ser.readline()
                parsed = parse_line(line)
                if parsed is None:
                    continue
                x, y, z = parsed
                elapsed = time.time() - start_time
                now_str = datetime.now().isoformat(timespec="milliseconds")
                if prev is not None:
                    dx, dy, dz = x - prev[0], y - prev[1], z - prev[2]
                    l2 = dx * dx + dy * dy + dz * dz
                else:
                    dx, dy, dz, l2 = 0.0, 0.0, 0.0, 0.0
                prev = (x, y, z)
                csv_writer.writerow([now_str, f"{elapsed:.3f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}",
                                     f"{dx:.4f}", f"{dy:.4f}", f"{dz:.4f}", f"{l2:.6f}"])
                sample_count[0] += 1
                if sample_count[0] % 100 == 0:
                    csv_file.flush()
                print(f"  x={x:+.3f} g   y={y:+.3f} g   z={z:+.3f} g   L2={l2:.6f}", end="\r")
        except KeyboardInterrupt:
            pass
        finally:
            csv_file.close()
            ser.close()
            print(f"\n\nDone. {sample_count[0]} samples saved to {csv_filename}")
        return

    # ------------------------------------------------------------------
    # Mode B: Live plot + CSV  (3 subplots: raw, deltas, L2 norm)
    # ------------------------------------------------------------------
    fig, (ax_raw, ax_delta, ax_l2) = plt.subplots(3, 1, figsize=(12, 9),
                                                    gridspec_kw={"height_ratios": [2, 2, 1.5]})
    fig.canvas.manager.set_window_title("MMA8452Q Live Accelerometer")

    # --- Subplot 1: Raw X, Y, Z ---
    line_x, = ax_raw.plot([], [], label="X", color="#e74c3c", linewidth=1.2)
    line_y, = ax_raw.plot([], [], label="Y", color="#2ecc71", linewidth=1.2)
    line_z, = ax_raw.plot([], [], label="Z", color="#3498db", linewidth=1.2)
    ax_raw.set_ylabel("Acceleration (g)")
    ax_raw.set_title("Raw Accelerometer Data")
    ax_raw.legend(loc="upper right")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.set_ylim(-2.5, 2.5)

    # --- Subplot 2: Deltas (dx, dy, dz) ---
    line_dx, = ax_delta.plot([], [], label="ΔX", color="#f97316", linewidth=1.0)
    line_dy, = ax_delta.plot([], [], label="ΔY", color="#a855f7", linewidth=1.0)
    line_dz, = ax_delta.plot([], [], label="ΔZ", color="#06b6d4", linewidth=1.0)
    ax_delta.set_ylabel("Δ Accel (g)")
    ax_delta.set_title("Point-to-Point Deltas")
    ax_delta.legend(loc="upper right")
    ax_delta.grid(True, alpha=0.3)
    ax_delta.set_ylim(-1.0, 1.0)

    # --- Subplot 3: L2 Norm ---
    line_l2, = ax_l2.plot([], [], label="L2 Norm", color="#6366f1", linewidth=1.5)
    ax_l2.axhline(y=0.002, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7, label="Threshold (0.002)")
    ax_l2.set_xlabel("Time (s)")
    ax_l2.set_ylabel("L2 Norm")
    ax_l2.set_title("L2 Norm (dx² + dy² + dz²)")
    ax_l2.legend(loc="upper right")
    ax_l2.grid(True, alpha=0.3)
    ax_l2.set_ylim(0, 0.05)

    def animate(_frame):
        read_and_store()
        if len(t_data) < 2:
            return (line_x, line_y, line_z, line_dx, line_dy, line_dz, line_l2)

        t = list(t_data)
        xlim = (t_data[0], t_data[-1] + 0.1)

        # Raw
        line_x.set_data(t, list(x_data))
        line_y.set_data(t, list(y_data))
        line_z.set_data(t, list(z_data))
        ax_raw.set_xlim(*xlim)

        # Deltas
        line_dx.set_data(t, list(dx_data))
        line_dy.set_data(t, list(dy_data))
        line_dz.set_data(t, list(dz_data))
        ax_delta.set_xlim(*xlim)

        # L2 Norm (auto-scale y if values exceed current limit)
        l2_list = list(l2_data)
        line_l2.set_data(t, l2_list)
        ax_l2.set_xlim(*xlim)
        max_l2 = max(l2_list) if l2_list else 0.05
        ax_l2.set_ylim(0, max(0.05, max_l2 * 1.2))

        return (line_x, line_y, line_z, line_dx, line_dy, line_dz, line_l2)

    ani = animation.FuncAnimation(fig, animate, interval=PLOT_INTERVAL_MS, blit=False, cache_frame_data=False)

    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        csv_file.close()
        ser.close()
        print(f"\nDone. {sample_count[0]} samples saved to {csv_filename}")


if __name__ == "__main__":
    main()
