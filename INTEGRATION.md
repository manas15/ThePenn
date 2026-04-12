# ThePenn Integration Guide

## UI Overview (http://localhost:8767)

The notes interface has these sections:

```
┌─────────────────────────────────────────────┐
│                  ThePenn                     │
│          Live handwriting recognition       │
│                                             │
│  [Start Mock Stream]  [Start Live Pen]      │  ← Input source toggle
│          ● Receiving words from pen...      │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │ mitosis is the process of cell      │    │  ← Live word stream
│  │ division where one cell divides     │    │    (words appear one by one
│  │ into two identical daughter cells   │    │     as the pen writes them)
│  │ the stages are prophase metaphase   │    │
│  └─────────────────────────────────────┘    │
│                                   70 words  │
│                                             │
│  [subject: ___] [Save to Ara] [Flashcards]  │  ← Ara actions
│                 [Summary] [Quiz Me] [Clear]  │
│                                             │
│  ThePenn Study Assistant              ×     │
│  ┌─────────────────────────────────────┐    │  ← Ara agent response
│  │ I've created your new notes for     │    │    (appears after clicking
│  │ "biology":                          │    │     any Ara action button)
│  │ "Mitosis is the process of cell..." │    │
│  │ Let me know if you want flashcards! │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### Two input modes

| Button | What it does |
|---|---|
| **Start Mock Stream** | Plays fake biology words at ~3/sec for testing without hardware |
| **Start Live Pen** | Launches `pennference.py`, connects to Arduino, streams real recognized words |

Both modes feed words into the same notes display. You can switch between them.
Only one can be active at a time.

### Ara action buttons

| Button | What it does |
|---|---|
| **Save to Ara** | Sends all captured words to the Ara agent, which stores them by subject + date |
| **Flashcards** | Generates Q/A flashcards from the saved notes |
| **Summary** | Creates a bullet-point summary of key concepts |
| **Quiz Me** | Agent asks you questions about your notes |
| **Clear** | Clears the word display (does not delete saved notes from Ara) |

The **subject field** (left of Save to Ara) tags your notes — e.g. "biology", "history".
Ara stores notes per subject so you can request flashcards/summaries for specific subjects later.

---

## How the Live Pen → Notes UI → Ara pipeline works

```
Arduino + Pen  →  pennference.py (port 8768)  →  POST /api/word  →  notes_server.py (port 8767)  →  Browser UI  →  Ara Agent
  (serial)        (segmenter + classifier          (HTTP)            (WebSocket broadcast)           (notes.html)    (flashcards,
                   + dashboard)                                                                                       summaries,
                                                                                                                      quizzes)
```

pennference.py runs as its own server with a live dashboard. When it classifies a
word above the confidence threshold, it POSTs `{"word": "...", "confidence": 0.85}`
to `notes_server.py`'s `/api/word` endpoint, which broadcasts it to the notes UI.

## Quick Start

```bash
# Terminal 1: Start the notes server
export ARA_API_KEY=<your-key>
export ARA_RUNTIME_KEY=<your-key>
python3 notes_server.py              # http://localhost:8767

# Terminal 2: Start pennference
python3 pennference.py               # dashboard at http://localhost:8768
                                      # auto-forwards words to localhost:8767

# Open http://localhost:8767 in your browser for the notes UI
# Open http://localhost:8768 for the pennference pipeline dashboard
# Write with the pen — words appear in both UIs
```

## How Live Pen Streaming Works

1. `pennference.py` reads the Arduino serial port, runs SegmentationTCN + WordClassifier
2. Its dashboard at `http://localhost:8768` shows the live pipeline (accel chart, P(writing), word queue)
3. When a word is classified above the confidence threshold, pennference POSTs to `notes_server.py`:
   ```
   POST http://localhost:8767/api/word
   {"word": "mitosis", "confidence": 0.85}
   ```
4. `notes_server.py` broadcasts `{"type": "word", "word": "mitosis", "confidence": "85%"}` to all connected `/ws/pen` WebSocket clients
5. The browser's `addWord()` function appends it to the notes display

The "Start Live Pen" button in the notes UI can also launch pennference as a subprocess (legacy mode).

## Adding Words Programmatically

### Option A: POST to `/api/word` (recommended)

```bash
curl -X POST http://localhost:8767/api/word \
  -H "Content-Type: application/json" \
  -d '{"word":"hello","confidence":0.9}'
```

This is what pennference.py uses internally. Any source can push words this way.

### Option B: Call `window.pushWord()` from JavaScript

In the browser console or from your own script loaded in the page:

```javascript
window.pushWord("hello");   // adds "hello" to the notes
window.pushWord("world");   // adds "world"
```

### Option C: Connect via WebSocket

Connect to `ws://localhost:8767/ws/pen` — words from `/api/word` are broadcast to all connected clients.

## Ara Agent

The Ara agent (`ara_app.py`) provides:

- **save_notes(text, subject, date)** — stores notes in Ara's cloud
- **get_notes(subject, date)** — retrieves stored notes
- **list_subjects()** — lists all subjects

Actions available from the UI:
- **Save to Ara** — sends all captured words to the agent for storage
- **Flashcards** — generates Q/A pairs from saved notes
- **Summary** — creates a structured summary
- **Quiz Me** — asks questions about the notes

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ARA_API_KEY` | Yes | Your Ara API key |
| `ARA_RUNTIME_KEY` | Yes | From `ara deploy ara_app.py` output |
| `PORT` | No | Server port (default: 8767) |

## File Structure

```
ThePenn/
├── notes_server.py          # Notes UI server (port 8767) + Ara + /api/word endpoint
├── pennference.py           # Inference pipeline + dashboard (port 8768), POSTs to notes_server
├── ara_app.py               # Ara agent definition (tools + study agent)
├── training/static/
│   ├── notes.html           # Notes browser UI
│   └── pennference.html     # Pennference pipeline dashboard
├── segmenter.pt             # Segmentation model weights
└── models/
    └── word_classifier_best.pt  # Classifier model weights
```
