"""
server/bridge.py

SerialBridge: reads accelerometer data in a background thread,
buffers it for live streaming and recording.
"""

import asyncio
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone

from serial_port import find_port, open_connection, parse_line


class SerialBridge:
    """Reads serial data in a background thread, buffers for streaming and recording."""

    def __init__(self, port=None):
        self.port = port
        self.loop = None
        self.connected = False
        self.running = True
        self.sample_rate = 0.0

        self.ring = deque(maxlen=200)

        self._lock = threading.Lock()
        self.recording = False
        self._rec_buffer = []
        self._rec_start = 0.0
        self._rec_wall_clock = ""

        self.session_active = False
        self.session_buffer = []
        self.session_start = 0.0
        self.session_wall_clock = ""
        self.session_events = []

        self._subscribers = set()

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        ser = None
        rate_count = 0
        rate_start = time.time()

        while self.running:
            if ser is None:
                try:
                    port = self.port or find_port()
                    if port is None:
                        self._broadcast_status(False)
                        time.sleep(2)
                        continue
                    ser = open_connection(port)
                    self.port = port
                    self._broadcast_status(True)
                    rate_count = 0
                    rate_start = time.time()
                except Exception:
                    self._broadcast_status(False)
                    time.sleep(2)
                    continue

            try:
                raw = ser.readline()
                if not raw:
                    continue
                parsed = parse_line(raw)
                if parsed is None:
                    continue

                x, y, z = parsed
                t = time.time()

                rate_count += 1
                elapsed = t - rate_start
                if elapsed >= 2.0:
                    self.sample_rate = round(rate_count / elapsed, 1)
                    rate_count = 0
                    rate_start = t

                self.ring.append((x, y, z, t))

                with self._lock:
                    if self.recording:
                        self._rec_buffer.append((x, y, z, t))
                    if self.session_active:
                        self.session_buffer.append((x, y, z, t, 1 if self.recording else 0))

                self._broadcast_accel(x, y, z, t)

            except Exception:
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                self._broadcast_status(False)
                time.sleep(1)

    def _broadcast_status(self, connected):
        self.connected = connected
        msg = json.dumps({
            "type": "status",
            "connected": connected,
            "port": self.port or "",
            "sample_rate_hz": self.sample_rate,
        })
        self._push_to_subscribers(msg)

    def _broadcast_accel(self, x, y, z, t):
        if not self.loop or not self._subscribers:
            return
        msg = json.dumps({"type": "accel", "x": x, "y": y, "z": z, "t": t})
        self._push_to_subscribers(msg)

    def _push_to_subscribers(self, msg):
        if not self.loop:
            return
        for q in list(self._subscribers):
            try:
                self.loop.call_soon_threadsafe(q.put_nowait, msg)
            except (asyncio.QueueFull, RuntimeError):
                pass

    def subscribe(self):
        q = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q):
        self._subscribers.discard(q)

    def start_recording(self):
        with self._lock:
            self.recording = True
            self._rec_buffer = []
            self._rec_start = time.time()
            self._rec_wall_clock = datetime.now(timezone.utc).isoformat()

    def stop_recording(self):
        with self._lock:
            self.recording = False
            buf = list(self._rec_buffer)
            start = self._rec_start
            wall_clock = self._rec_wall_clock
        return buf, start, wall_clock

    def start_session(self):
        with self._lock:
            self.session_active = True
            self.session_buffer = []
            self.session_events = []
            self.session_start = time.time()
            self.session_wall_clock = datetime.now(timezone.utc).isoformat()

    def stop_session(self):
        with self._lock:
            self.session_active = False

    def finalize_session(self):
        with self._lock:
            buf = list(self.session_buffer)
            events = list(self.session_events)
            start = self.session_start
            wall_clock = self.session_wall_clock
            self.session_buffer = []
            self.session_events = []
        return buf, start, wall_clock, events

    def add_session_event(self, event):
        with self._lock:
            self.session_events.append(event)

    def stop(self):
        self.running = False
