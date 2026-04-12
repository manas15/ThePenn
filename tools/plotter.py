"""
tools/plotter.py

Live accelerometer plotter + CSV logger.

Usage:
    python -m tools.plotter
    python -m tools.plotter --port /dev/cu.usbmodem2101
    python -m tools.plotter --no-plot
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime

import serial.tools.list_ports
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serial_port import find_port, open_connection, parse_line

MAX_POINTS = 200
PLOT_INTERVAL_MS = 50


def main():
    parser = argparse.ArgumentParser(description="Live accelerometer plotter")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (auto-detected if omitted)")
    parser.add_argument("--no-plot", action="store_true",
                        help="CSV logging only, no plot")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for CSV output")
    args = parser.parse_args()

    port = args.port or find_port()
    if port is None:
        print("ERROR: No Arduino found. Plug it in or specify --port.")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}  -  {p.description}")
        sys.exit(1)

    print(f"Connecting to {port}...")
    ser = open_connection(port)
    print("Connected!\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"accel_log_{ts}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "elapsed_s", "x_g", "y_g", "z_g", "dx", "dy", "dz", "l2_norm"])
    print(f"Logging to: {csv_path}\n")

    t_data = deque(maxlen=MAX_POINTS)
    x_data = deque(maxlen=MAX_POINTS)
    y_data = deque(maxlen=MAX_POINTS)
    z_data = deque(maxlen=MAX_POINTS)
    dx_data = deque(maxlen=MAX_POINTS)
    dy_data = deque(maxlen=MAX_POINTS)
    dz_data = deque(maxlen=MAX_POINTS)
    l2_data = deque(maxlen=MAX_POINTS)
    prev_reading = [None]
    start_time = time.time()
    sample_count = [0]

    def read_and_store():
        while ser.in_waiting:
            line = ser.readline()
            parsed = parse_line(line)
            if parsed is None:
                continue
            x, y, z = parsed
            elapsed = time.time() - start_time
            now_str = datetime.now().isoformat(timespec="milliseconds")

            if prev_reading[0] is not None:
                px, py, pz = prev_reading[0]
                dx, dy, dz = x - px, y - py, z - pz
                l2 = dx * dx + dy * dy + dz * dz
            else:
                dx, dy, dz, l2 = 0.0, 0.0, 0.0, 0.0
            prev_reading[0] = (x, y, z)

            csv_writer.writerow([now_str, f"{elapsed:.3f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}",
                                 f"{dx:.4f}", f"{dy:.4f}", f"{dz:.4f}", f"{l2:.6f}"])

            t_data.append(elapsed)
            x_data.append(x); y_data.append(y); z_data.append(z)
            dx_data.append(dx); dy_data.append(dy); dz_data.append(dz)
            l2_data.append(l2)

            sample_count[0] += 1
            if sample_count[0] % 100 == 0:
                csv_file.flush()

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
            print(f"\n\nDone. {sample_count[0]} samples saved to {csv_path}")
        return

    fig, (ax_raw, ax_delta, ax_l2) = plt.subplots(3, 1, figsize=(12, 9),
                                                    gridspec_kw={"height_ratios": [2, 2, 1.5]})
    fig.canvas.manager.set_window_title("ThePenn Live Accelerometer")

    line_x, = ax_raw.plot([], [], label="X", color="#e74c3c", linewidth=1.2)
    line_y, = ax_raw.plot([], [], label="Y", color="#2ecc71", linewidth=1.2)
    line_z, = ax_raw.plot([], [], label="Z", color="#3498db", linewidth=1.2)
    ax_raw.set_ylabel("Acceleration (g)")
    ax_raw.set_title("Raw Accelerometer Data")
    ax_raw.legend(loc="upper right")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.set_ylim(-2.5, 2.5)

    line_dx, = ax_delta.plot([], [], label="DX", color="#f97316", linewidth=1.0)
    line_dy, = ax_delta.plot([], [], label="DY", color="#a855f7", linewidth=1.0)
    line_dz, = ax_delta.plot([], [], label="DZ", color="#06b6d4", linewidth=1.0)
    ax_delta.set_ylabel("Delta Accel (g)")
    ax_delta.set_title("Point-to-Point Deltas")
    ax_delta.legend(loc="upper right")
    ax_delta.grid(True, alpha=0.3)
    ax_delta.set_ylim(-1.0, 1.0)

    line_l2, = ax_l2.plot([], [], label="L2 Norm", color="#6366f1", linewidth=1.5)
    ax_l2.axhline(y=0.002, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7, label="Threshold")
    ax_l2.set_xlabel("Time (s)")
    ax_l2.set_ylabel("L2 Norm")
    ax_l2.set_title("L2 Norm (dx^2 + dy^2 + dz^2)")
    ax_l2.legend(loc="upper right")
    ax_l2.grid(True, alpha=0.3)
    ax_l2.set_ylim(0, 0.05)

    def animate(_frame):
        read_and_store()
        if len(t_data) < 2:
            return (line_x, line_y, line_z, line_dx, line_dy, line_dz, line_l2)

        t = list(t_data)
        xlim = (t_data[0], t_data[-1] + 0.1)

        line_x.set_data(t, list(x_data))
        line_y.set_data(t, list(y_data))
        line_z.set_data(t, list(z_data))
        ax_raw.set_xlim(*xlim)

        line_dx.set_data(t, list(dx_data))
        line_dy.set_data(t, list(dy_data))
        line_dz.set_data(t, list(dz_data))
        ax_delta.set_xlim(*xlim)

        l2_list = list(l2_data)
        line_l2.set_data(t, l2_list)
        ax_l2.set_xlim(*xlim)
        max_l2 = max(l2_list) if l2_list else 0.05
        ax_l2.set_ylim(0, max(0.05, max_l2 * 1.2))

        return (line_x, line_y, line_z, line_dx, line_dy, line_dz, line_l2)

    _ani = animation.FuncAnimation(fig, animate, interval=PLOT_INTERVAL_MS, blit=False, cache_frame_data=False)

    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        csv_file.close()
        ser.close()
        print(f"\nDone. {sample_count[0]} samples saved to {csv_path}")


if __name__ == "__main__":
    main()
