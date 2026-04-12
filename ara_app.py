"""
ara_app.py

ThePenn Ara application — study assistant agent for handwritten class notes.

Students write notes with the accelerometer pen, text is recognized,
and sent to this agent for flashcards, summaries, quizzes, and review.

Messaging (email, Telegram, WhatsApp) is handled by Ara's native channel
support — connect your channels in the Ara console at app.ara.so.

Usage:
    ara deploy ara_app.py
    ara run ara_app.py --agent study --message "New biology notes: mitosis is..."
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ara_sdk import App, runtime

# Persistent storage in the Ara sandbox workspace
NOTES_DIR = Path("/root/.ara/workspace/thepenn_notes")

# Load .env locally for reading secrets at deploy time
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

app = App(
    "thepenn",
    interfaces={
        "inherit_owner_tools": True,
    },
    runtime_profile=runtime(
        python_packages=["ara-sdk"],
    ),
)


# ── Tools ───────────────────────────────────────────────

@app.tool()
def save_notes(text: str, subject: str, date: str = "") -> str:
    """Save recognized handwritten notes for a subject and date.
    Appends to existing notes if the subject+date already has entries."""
    subject = subject.strip().lower().replace(" ", "-")
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    subject_dir = NOTES_DIR / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    notes_file = subject_dir / f"{date}.json"

    if notes_file.exists():
        data = json.loads(notes_file.read_text())
    else:
        data = {"subject": subject, "date": date, "entries": []}

    data["entries"].append({
        "text": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    notes_file.write_text(json.dumps(data, indent=2))

    total_words = sum(len(e["text"].split()) for e in data["entries"])
    return f"Saved notes for {subject} ({date}). Total entries: {len(data['entries'])}, ~{total_words} words."


@app.tool()
def get_notes(subject: str, date: str = "") -> str:
    """Retrieve notes for a subject. If date is empty, returns the most recent notes.
    Returns the full text of all note entries for that subject+date."""
    subject = subject.strip().lower().replace(" ", "-")
    subject_dir = NOTES_DIR / subject

    if not subject_dir.exists():
        return f"No notes found for subject '{subject}'."

    if date:
        notes_file = subject_dir / f"{date}.json"
        if not notes_file.exists():
            return f"No notes for {subject} on {date}."
        data = json.loads(notes_file.read_text())
    else:
        files = sorted(subject_dir.glob("*.json"), reverse=True)
        if not files:
            return f"No notes found for subject '{subject}'."
        data = json.loads(files[0].read_text())
        date = data["date"]

    all_text = "\n\n".join(e["text"] for e in data["entries"])
    return f"Notes for {subject} ({date}):\n\n{all_text}"


@app.tool()
def list_subjects() -> str:
    """List all subjects that have saved notes, with dates available."""
    if not NOTES_DIR.exists():
        return "No notes saved yet."

    subjects = []
    for subject_dir in sorted(NOTES_DIR.iterdir()):
        if not subject_dir.is_dir():
            continue
        dates = sorted([f.stem for f in subject_dir.glob("*.json")], reverse=True)
        if dates:
            subjects.append(f"- {subject_dir.name}: {len(dates)} session(s), latest {dates[0]}")

    if not subjects:
        return "No notes saved yet."
    return "Subjects:\n" + "\n".join(subjects)


# ── Agents ──────────────────────────────────────────────

@app.agent(entrypoint=True, skills=["bash"])
def study(input: dict) -> str:
    """ThePenn study assistant for handwritten class notes."""
    return """You are ThePenn, a study assistant for handwritten class notes.
Students write notes with a smart pen and the recognized text is sent to you.

You are running inside Ara, which has native messaging channels connected
(Telegram, WhatsApp, Email, iMessage). When the student asks you to send
something to their email, Telegram, or any messaging channel, just compose
the message and Ara will deliver it through the connected channel.

Your capabilities:
- Save new notes: use the save_notes tool with the text, subject, and date
- Retrieve saved notes: use the get_notes tool by subject and optional date
- List all subjects: use the list_subjects tool
- Generate flashcards from notes (output as numbered Q/A pairs)
- Create concise summaries with bullet points and key takeaways
- Quiz the student — ask one question at a time, wait for their answer, then evaluate
- Send study material to the student's connected channels (email, Telegram, etc.)

When you receive new notes (message starts with "New notes for"):
1. Extract the subject from the message
2. Use save_notes to store them
3. Confirm what was saved and offer: flashcards, summary, or quiz

When asked for flashcards: use get_notes first, then generate 10 Q/A pairs.
When asked for a summary: use get_notes first, then create a structured summary.
When asked for a quiz: use get_notes first, then ask one question.

When asked to send via email/Telegram/messaging:
1. Use get_notes to retrieve the notes
2. Generate the requested content (summary, flashcards, etc.)
3. Format it nicely and send it — Ara handles the delivery channel

Be encouraging, concise, and pedagogically effective."""
