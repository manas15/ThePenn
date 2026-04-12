# ThePenn Integration Guide

## How the Live Pen → Notes UI → Ara pipeline works

```
Arduino + Pen  →  pennference.py  →  notes_server.py  →  Browser UI  →  Ara Agent
  (serial)        (segmenter +       (WebSocket          (notes.html)    (flashcards,
                   classifier)        bridge)                             summaries,
                                                                         quizzes)
```

## Quick Start

```bash
# Terminal 1: Start the notes server
export ARA_API_KEY=<your-key>
export ARA_RUNTIME_KEY=<your-key>
python3 notes_server.py

# Open http://localhost:8767 in your browser
# Click "Start Live Pen" — this launches pennference.py in the background
# Write with the pen — recognized words appear in real time
# Click "Save to Ara" to send notes to the study agent
```

## How Live Pen Streaming Works

When the user clicks **"Start Live Pen"** in the browser:

1. Browser opens a WebSocket to `ws://localhost:8767/ws/pen`
2. Server launches `pennference.py` as a subprocess
3. `pennference.py` reads the Arduino serial port, runs the segmenter + classifier
4. Server parses stdout from pennference.py looking for recognized words
5. Each recognized word is pushed to the browser via WebSocket as:
   ```json
   {"type": "word", "word": "mitosis", "confidence": "85%"}
   ```
6. The browser's `addWord()` function appends it to the notes display

## pennference.py Output Format

The server parses these patterns from pennference.py stdout:

| stdout pattern | What it means |
|---|---|
| `WORD STARTED` | Pen started writing (segmenter detected motion) |
| `WORD IS: <word>  (<confidence>%)` | A word was classified above the confidence threshold |
| `LOW CONFIDENCE: <word>  (<confidence>%) -- skipped` | Classification below threshold |
| `>>> word1 word2 word3` | Accumulated sentence so far |

## Adding Words Programmatically

If you want to push words from a different source (e.g. a custom classifier, a different serial reader), you have two options:

### Option A: Call `window.pushWord()` from JavaScript

In the browser console or from your own script loaded in the page:

```javascript
window.pushWord("hello");   // adds "hello" to the notes
window.pushWord("world");   // adds "world"
```

### Option B: Send words via WebSocket

Connect to `ws://localhost:8767/ws/pen` and the server will forward words.
Or modify `notes_server.py` to accept words from your own WebSocket/HTTP endpoint.

### Option C: POST words via HTTP (simple)

If you prefer HTTP, add a route to `notes_server.py`:

```python
async def api_push_word(request):
    data = await request.json()
    word = data.get("word", "")
    # Broadcast to connected WebSocket clients
    ...
```

Then call it:
```bash
curl -X POST http://localhost:8767/api/word -H "Content-Type: application/json" -d '{"word":"hello"}'
```

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
├── notes_server.py          # Web server (UI + Ara + pen WebSocket)
├── ara_app.py               # Ara agent definition (tools + study agent)
├── pennference.py           # Ryan's inference pipeline (serial → words)
├── training/static/
│   └── notes.html           # Browser UI
├── segmenter.pt             # Segmentation model weights
└── models/
    └── word_classifier_best.pt  # Classifier model weights
```
