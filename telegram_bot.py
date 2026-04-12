"""
telegram_bot.py

Telegram bot bridge for ThePenn Ara agent.
Forwards messages from Telegram to the Ara study agent and returns responses.

Setup:
    1. Create a bot via @BotFather on Telegram, get the token
    2. Add TELEGRAM_BOT_TOKEN to your .env file
    3. Run: python telegram_bot.py

Requirements:
    pip install python-telegram-bot
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

PROJECT_ROOT = str(Path(__file__).resolve().parent)
ARA_APP_PATH = str(Path(__file__).resolve().parent / "ara_app.py")

# Load .env
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
if not TELEGRAM_BOT_TOKEN:
    print("Error: TELEGRAM_BOT_TOKEN not set in .env")
    sys.exit(1)


async def run_ara_agent(message: str) -> str:
    """Send a message to the Ara study agent and return the response."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "ara_sdk",
        "run", ARA_APP_PATH,
        "--agent", "study",
        "--message", message,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode().strip()[-500:]
        return f"Ara error: {err}"

    try:
        result = json.loads(stdout.decode())
        return result.get("result", {}).get("output_text", "No response from Ara.")
    except json.JSONDecodeError:
        return f"Unexpected response: {stdout.decode()[:500]}"


# -- Command handlers --

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hey! I'm ThePenn, your study assistant.\n\n"
        "Send me your notes and I'll save them. You can also ask me to:\n"
        "/save <subject> - save the last notes you sent\n"
        "/flashcards <subject> - generate flashcards\n"
        "/summary <subject> - summarize your notes\n"
        "/quiz <subject> - quiz you on your notes\n"
        "/subjects - list all saved subjects\n\n"
        "Or just chat with me about your notes!"
    )


async def cmd_subjects(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Looking up your subjects...")
    response = await run_ara_agent("List all my saved subjects")
    await update.message.reply_text(response)


async def cmd_flashcards(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subject = " ".join(context.args) if context.args else "general"
    await update.message.reply_text(f"Generating flashcards for {subject}...")
    response = await run_ara_agent(f"Generate flashcards for my {subject} notes")
    # Split long messages (Telegram limit is 4096 chars)
    for chunk in _split_message(response):
        await update.message.reply_text(chunk)


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subject = " ".join(context.args) if context.args else "general"
    await update.message.reply_text(f"Summarizing {subject} notes...")
    response = await run_ara_agent(f"Summarize my {subject} notes")
    for chunk in _split_message(response):
        await update.message.reply_text(chunk)


async def cmd_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subject = " ".join(context.args) if context.args else "general"
    await update.message.reply_text(f"Let me quiz you on {subject}...")
    response = await run_ara_agent(f"Quiz me on my {subject} notes")
    await update.message.reply_text(response)


async def cmd_save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subject = " ".join(context.args) if context.args else "general"
    last_notes = context.user_data.get("last_notes", "")
    if not last_notes:
        await update.message.reply_text(
            "No recent notes to save. Send me some notes first, then use /save <subject>"
        )
        return
    await update.message.reply_text(f"Saving your notes under '{subject}'...")
    response = await run_ara_agent(f"New notes for {subject}: {last_notes}")
    context.user_data["last_notes"] = ""
    await update.message.reply_text(response)


# -- Message handler (free-form text) --

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text.strip()
    if not text:
        return

    # Store as potential notes for /save
    context.user_data["last_notes"] = text

    await update.message.reply_text("Thinking...")
    response = await run_ara_agent(text)
    for chunk in _split_message(response):
        await update.message.reply_text(chunk)


def _split_message(text: str, max_len: int = 4096) -> list[str]:
    """Split a message into chunks that fit Telegram's character limit."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Try to split at a newline
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


def main() -> None:
    print(f"Starting ThePenn Telegram bot...")
    print(f"Ara app: {ARA_APP_PATH}")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_start))
    application.add_handler(CommandHandler("subjects", cmd_subjects))
    application.add_handler(CommandHandler("flashcards", cmd_flashcards))
    application.add_handler(CommandHandler("summary", cmd_summary))
    application.add_handler(CommandHandler("quiz", cmd_quiz))
    application.add_handler(CommandHandler("save", cmd_save))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running! Send /start in Telegram to begin.")
    application.run_polling()


if __name__ == "__main__":
    main()
