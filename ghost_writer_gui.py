"""
ghost_writer_gui.py

Live GUI that reads MMA8452Q accelerometer data from an Arduino Uno
and displays a fullscreen indicator:

    GREEN + "WRITING"       when the pen is moving
    RED   + "NOT WRITING"   when the pen is still

The L2 norm of the acceleration deltas (dx² + dy² + dz²) is computed
live from consecutive samples. If it exceeds a threshold the sensor
is considered to be "writing".

Requirements:
    pip install pyserial

Usage:
    python ghost_writer_gui.py                         # auto-detect port
    python ghost_writer_gui.py --port /dev/cu.usbmodem2101
    python ghost_writer_gui.py --threshold 0.005       # adjust sensitivity
"""

import argparse
import sys
import time
import threading
import tkinter as tk

import serial
import serial.tools.list_ports


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
BAUD_RATE = 9600
DEFAULT_THRESHOLD = 0.002   # L2 norm cutoff (squared delta magnitude)
SMOOTHING_WINDOW = 3        # average over N samples to reduce flicker


def find_arduino_port():
    """Auto-detect the first likely Arduino serial port."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        mfr = (p.manufacturer or "").lower()
        if any(kw in desc for kw in ["arduino", "ch340", "usb serial", "acm"]):
            return p.device
        if "arduino" in mfr:
            return p.device
    if len(ports) == 1:
        return ports[0].device
    return None


def parse_line(raw):
    """Parse 'X\\tY\\tZ' from serial. Returns (x, y, z) or None."""
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


class GhostWriterApp:
    def __init__(self, root, port, threshold, smoothing):
        self.root = root
        self.threshold = threshold
        self.smoothing = smoothing

        # --- State ---
        self.prev_reading = None
        self.recent_l2 = []           # rolling window of L2 values
        self.is_writing = False
        self.current_l2 = 0.0
        self.running = True

        # --- Serial ---
        self.ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)
        self.ser.reset_input_buffer()

        # --- GUI setup ---
        self.root.title("Ghost Writer")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<q>", lambda e: self.quit())

        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Main status text
        self.status_text = self.canvas.create_text(
            0, 0, text="NOT WRITING", fill="white",
            font=("Helvetica", 96, "bold"), anchor="center"
        )

        # L2 value readout (smaller, bottom)
        self.l2_text = self.canvas.create_text(
            0, 0, text="L2: 0.000000", fill="white",
            font=("Courier", 20), anchor="center"
        )

        # Threshold label
        self.thresh_text = self.canvas.create_text(
            0, 0, text="threshold: " + str(self.threshold), fill="white",
            font=("Courier", 14), anchor="center"
        )

        self.canvas.bind("<Configure>", self.on_resize)

        # --- Start serial reader thread ---
        self.thread = threading.Thread(target=self.serial_loop, daemon=True)
        self.thread.start()

        # --- Start GUI update loop ---
        self.update_gui()

    def on_resize(self, event):
        w, h = event.width, event.height
        self.canvas.coords(self.status_text, w // 2, h // 2 - 20)
        self.canvas.coords(self.l2_text, w // 2, h // 2 + 80)
        self.canvas.coords(self.thresh_text, w // 2, h - 40)

    def serial_loop(self):
        """Background thread: read serial data and compute L2 norm."""
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
                    dx = x - px
                    dy = y - py
                    dz = z - pz
                    l2 = dx * dx + dy * dy + dz * dz

                    self.recent_l2.append(l2)
                    if len(self.recent_l2) > self.smoothing:
                        self.recent_l2.pop(0)

                    avg_l2 = sum(self.recent_l2) / len(self.recent_l2)
                    self.current_l2 = avg_l2
                    self.is_writing = avg_l2 >= self.threshold

                self.prev_reading = (x, y, z)

            except Exception:
                if not self.running:
                    break
                time.sleep(0.1)

    def update_gui(self):
        """Periodically refresh the GUI from the main thread."""
        if not self.running:
            return

        if self.is_writing:
            bg = "#16a34a"      # green
            label = "WRITING"
        else:
            bg = "#dc2626"      # red
            label = "NOT WRITING"

        self.canvas.configure(bg=bg)
        self.canvas.itemconfig(self.status_text, text=label)
        self.canvas.itemconfig(self.l2_text,
                               text="L2: " + format(self.current_l2, ".6f"))

        self.root.after(50, self.update_gui)    # ~20 fps refresh

    def quit(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(
        description="Ghost Writer – live writing detection GUI")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (auto-detected if omitted)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="L2 norm cutoff for writing detection "
                             "(default: %(default)s)")
    parser.add_argument("--smoothing", type=int, default=SMOOTHING_WINDOW,
                        help="Number of samples to average for smoothing "
                             "(default: %(default)s)")
    args = parser.parse_args()

    port = args.port or find_arduino_port()
    if port is None:
        print("ERROR: No Arduino found. Plug it in or specify --port.")
        for p in serial.tools.list_ports.comports():
            print("  " + p.device + "  –  " + p.description)
        sys.exit(1)

    print("Connecting to " + port + " at " + str(BAUD_RATE) + " baud ...")
    root = tk.Tk()
    app = GhostWriterApp(root, port, args.threshold, args.smoothing)
    print("Running! Press Escape or Q to exit.")
    root.mainloop()


if __name__ == "__main__":
    main()
