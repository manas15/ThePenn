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
def save_notes(text: str, subject: str) -> str:
    """Save handwritten notes by writing them as a markdown file in the Ara workspace.
    This makes notes visible in the Ara console and accessible to the agent's memory."""
    import os
    from datetime import datetime, timezone

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    time_str = datetime.now(timezone.utc).strftime("%H:%M")
    notes_dir = "/root/.ara/workspace"
    filepath = os.path.join(notes_dir, f"notes-{subject}.md")

    header = f"# {subject.title()} Notes\n\n"
    entry = f"## {date} {time_str}\n\n{text}\n\n---\n\n"

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing = f.read()
        content = existing + entry
    else:
        content = header + entry

    with open(filepath, "w") as f:
        f.write(content)

    word_count = len(text.split())
    return f"Saved {word_count} words of {subject} notes to {filepath}"


@app.tool()
def get_notes(subject: str) -> str:
    """Read saved notes for a subject from the Ara workspace."""
    import os

    filepath = os.path.join("/root/.ara/workspace", f"notes-{subject}.md")
    if not os.path.exists(filepath):
        return f"No notes found for '{subject}'."
    with open(filepath, "r") as f:
        return f.read()


@app.tool()
def list_subjects() -> str:
    """List all subjects that have saved notes."""
    import os
    import glob

    files = glob.glob("/root/.ara/workspace/notes-*.md")
    if not files:
        return "No notes saved yet."
    subjects = []
    for f in sorted(files):
        name = os.path.basename(f).replace("notes-", "").replace(".md", "")
        size = os.path.getsize(f)
        subjects.append(f"- {name} ({size} bytes)")
    return "Saved subjects:\n" + "\n".join(subjects)


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
2. Use save_notes tool to store them — you MUST actually call the tool, do not skip it
3. After the tool confirms, offer: flashcards, summary, or quiz

When asked for flashcards: call get_notes tool first, then generate 10 Q/A pairs.
When asked for a summary: call get_notes tool first, then create a structured summary.
When asked for a quiz: call get_notes tool first, then ask one question.

CRITICAL: Always call the tools. Never pretend to have called a tool. If a tool fails, report the error.

When asked to send via email/Telegram/messaging:
1. Use get_notes to retrieve the notes
2. Generate the requested content (summary, flashcards, etc.)
3. Format it nicely and send it — Ara handles the delivery channel

Be encouraging, concise, and pedagogically effective."""
