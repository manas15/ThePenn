"""
notes_server.py

Lightweight server for the ThePenn notes UI + Ara integration.
No Arduino/serial dependencies — just the web UI and Ara agent.

Usage:
    export ARA_API_KEY=...
    export ARA_RUNTIME_KEY=...
    python3 notes_server.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from aiohttp import web

STATIC_DIR = Path(__file__).resolve().parent / "training" / "static"
ARA_APP_PATH = str(Path(__file__).resolve().parent / "ara_app.py")


async def notes_page(request):
    # Serve with no-cache headers to avoid stale content
    return web.FileResponse(
        STATIC_DIR / "notes.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


async def static_handler(request):
    name = request.match_info["filename"]
    path = STATIC_DIR / name
    if path.exists() and path.is_file():
        return web.FileResponse(
            path,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )
    raise web.HTTPNotFound()


async def api_ara(request):
    data = await request.json()
    action = data.get("action", "save")
    subject = data.get("subject", "general").strip()
    text = data.get("text", "").strip()

    if not text:
        return web.json_response({"text": "No text provided.", "error": True})

    if action == "save":
        msg = f"New notes for {subject}: {text}"
    elif action == "flashcards":
        msg = f"Generate flashcards for my {subject} notes"
    elif action == "quiz":
        msg = f"Quiz me on my {subject} notes"
    elif action == "summary":
        msg = f"Summarize my {subject} notes"
    else:
        msg = text

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "ara_sdk",
            "run", ARA_APP_PATH,
            "--agent", "study",
            "--message", msg,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(__file__).resolve().parent),
            env=os.environ.copy(),
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode().strip()[-500:]
            print(f"[ara] error: {err}")
            return web.json_response({"text": f"Ara error: {err}", "error": True})

        result = json.loads(stdout.decode())
        output = result.get("result", {}).get("output_text", "No response.")
        print(f"[ara] {action}: {output[:120]}...")
        return web.json_response({"text": output, "action": action})
    except Exception as e:
        print(f"[ara] exception: {e}")
        return web.json_response({"text": f"Failed: {e}", "error": True})


app = web.Application()
app.router.add_get("/", notes_page)
app.router.add_get("/static/{filename}", static_handler)
app.router.add_post("/api/ara", api_ara)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8767"))
    print(f"ThePenn Notes UI: http://localhost:{port}")
    print(f"Ara app: {ARA_APP_PATH}")
    print()
    web.run_app(app, host="localhost", port=port, print=None)
