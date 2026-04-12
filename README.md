# ThePenn

Handwriting recognition system using an MMA8452Q accelerometer attached to a pen.
Captures 3-axis motion data via Arduino, streams it to a browser-based training UI,
and trains PyTorch models for word and character-level recognition.

## Architecture

```
firmware/       Arduino sketch for MMA8452Q accelerometer
server/         aiohttp WebSocket server + browser UI for data collection
ml/             PyTorch models and data pipeline
tools/          Standalone utilities (live plotter, writing detector GUI)
data/           Training data (JSONL samples, session recordings, audio)
```

## Quick Start

```bash
pip install -r requirements.txt

# Upload firmware/accelerometer.ino to your Arduino

# Start the training data collection server
python -m server

# Open http://localhost:8765 in your browser
```

## Standalone Tools

```bash
# Live accelerometer plot + CSV logging
python -m tools.plotter

# Fullscreen writing detection GUI
python -m tools.detector
```

## Hardware

- Arduino Uno (or compatible)
- SparkFun MMA8452Q 3-axis accelerometer breakout
- I2C connection with 330 Ohm series resistors (3.3V logic level)
