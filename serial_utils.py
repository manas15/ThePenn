"""
serial_utils.py

Shared serial communication utilities for the ghost-writer project.
Extracted from read_accelerometer.py and ghost_writer_gui.py.
"""

import time

import serial
import serial.tools.list_ports


BAUD_RATE = 9600


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


def open_serial(port, baud=BAUD_RATE):
    """Open the serial connection and wait for the Arduino to reset."""
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # Arduino resets on serial open; wait for it
    ser.reset_input_buffer()
    return ser


def parse_line(raw):
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
