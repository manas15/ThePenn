"""
serial_port.py

Serial communication utilities for reading accelerometer data from Arduino.
"""

import time

import serial
import serial.tools.list_ports

BAUD_RATE = 9600

ARDUINO_KEYWORDS = ("arduino", "ch340", "usb serial", "acm")


def find_port():
    """Auto-detect the first likely Arduino serial port."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        mfr = (p.manufacturer or "").lower()
        if any(kw in desc for kw in ARDUINO_KEYWORDS) or "arduino" in mfr:
            return p.device
    if len(ports) == 1:
        return ports[0].device
    return None


def open_connection(port, baud=BAUD_RATE):
    """Open a serial connection, waiting for Arduino reset."""
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    return ser


def parse_line(raw):
    """
    Parse a serial line: 'X\\tY\\tZ'.
    Returns (x, y, z) as floats, or None on invalid data.
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
