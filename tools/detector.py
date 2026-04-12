"""
tools/detector.py

Fullscreen GUI that shows WRITING / NOT WRITING based on accelerometer L2 norm.

Usage:
    python -m tools.detector
    python -m tools.detector --port /dev/cu.usbmodem2101
    python -m tools.detector --threshold 0.005
"""

import argparse
import os
import sys
import time
import threading
import tkinter as tk

import serial.tools.list_ports

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serial_port import find_port, open_connection, parse_line

DEFAULT_THRESHOLD = 0.002
DEFAULT_SMOOTHING = 3


class DetectorApp:
    def __init__(self, root, port, threshold, smoothing):
        self.root = root
        self.threshold = threshold
        self.smoothing = smoothing
        self.prev_reading = None
        self.recent_l2 = []
        self.is_writing = False
        self.current_l2 = 0.0
        self.running = True

        self.ser = open_connection(port)

        self.root.title("ThePenn Detector")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<q>", lambda e: self.quit())

        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_text = self.canvas.create_text(
            0, 0, text="NOT WRITING", fill="white",
            font=("Helvetica", 96, "bold"), anchor="center"
        )
        self.l2_text = self.canvas.create_text(
            0, 0, text="L2: 0.000000", fill="white",
            font=("Courier", 20), anchor="center"
        )
        self.thresh_text = self.canvas.create_text(
            0, 0, text=f"threshold: {self.threshold}", fill="white",
            font=("Courier", 14), anchor="center"
        )

        self.canvas.bind("<Configure>", self._on_resize)

        self._thread = threading.Thread(target=self._serial_loop, daemon=True)
        self._thread.start()
        self._update_gui()

    def _on_resize(self, event):
        w, h = event.width, event.height
        self.canvas.coords(self.status_text, w // 2, h // 2 - 20)
        self.canvas.coords(self.l2_text, w // 2, h // 2 + 80)
        self.canvas.coords(self.thresh_text, w // 2, h - 40)

    def _serial_loop(self):
        while self.running:
            try:
                line = self.ser.readline()
                if not line:
                    continue
                parsed = parse_line(line)
                if parsed is None:
                    continue

                x, y, z = parsed
                if self.prev_reading is not None:
                    px, py, pz = self.prev_reading
                    dx, dy, dz = x - px, y - py, z - pz
                    l2 = dx * dx + dy * dy + dz * dz
                    self.recent_l2.append(l2)
                    if len(self.recent_l2) > self.smoothing:
                        self.recent_l2.pop(0)
                    avg = sum(self.recent_l2) / len(self.recent_l2)
                    self.current_l2 = avg
                    self.is_writing = avg >= self.threshold
                self.prev_reading = (x, y, z)

            except Exception:
                if not self.running:
                    break
                time.sleep(0.1)

    def _update_gui(self):
        if not self.running:
            return
        if self.is_writing:
            bg, label = "#16a34a", "WRITING"
        else:
            bg, label = "#dc2626", "NOT WRITING"

        self.canvas.configure(bg=bg)
        self.canvas.itemconfig(self.status_text, text=label)
        self.canvas.itemconfig(self.l2_text, text=f"L2: {self.current_l2:.6f}")
        self.root.after(50, self._update_gui)

    def quit(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="ThePenn - live writing detection")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (auto-detected if omitted)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"L2 norm cutoff (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--smoothing", type=int, default=DEFAULT_SMOOTHING,
                        help=f"Smoothing window size (default: {DEFAULT_SMOOTHING})")
    args = parser.parse_args()

    port = args.port or find_port()
    if port is None:
        print("ERROR: No Arduino found. Plug it in or specify --port.")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}  -  {p.description}")
        sys.exit(1)

    print(f"Connecting to {port}...")
    root = tk.Tk()
    DetectorApp(root, port, args.threshold, args.smoothing)
    print("Running! Press Escape or Q to exit.")
    root.mainloop()


if __name__ == "__main__":
    main()
