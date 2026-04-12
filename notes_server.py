"""
notes_server.py

Lightweight server for the ThePenn notes UI + Ara integration.
Supports both mock data and live pen input via pennference.py.

Usage:
    python3 notes_server.py
    # Reads ARA_API_KEY and ARA_RUNTIME_KEY from .env file automatically
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from aiohttp import web

# Load .env file if it exists
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

STATIC_DIR = Path(__file__).resolve().parent / "training" / "static"
ARA_APP_PATH = str(Path(__file__).resolve().parent / "ara_app.py")
PENNFERENCE_PATH = str(Path(__file__).resolve().parent / "pennference.py")
PROJECT_ROOT = str(Path(__file__).resolve().parent)

NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"}


async def notes_page(request):
    return web.FileResponse(STATIC_DIR / "notes.html", headers=NO_CACHE)


async def static_handler(request):
    name = request.match_info["filename"]
    path = STATIC_DIR / name
    if path.exists() and path.is_file():
        return web.FileResponse(path, headers=NO_CACHE)
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
            cwd=PROJECT_ROOT,
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


async def ws_pen(request):
    """
    WebSocket endpoint for live pen streaming.
    Launches pennference.py as a subprocess, parses its stdout for recognized
    words, and pushes them to the browser in real time.
    """
    ws_resp = web.WebSocketResponse()
    await ws_resp.prepare(request)

    proc = None

    try:
        async for msg in ws_resp:
            if msg.type != web.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            cmd = data.get("type")

            if cmd == "start":
                if proc and proc.returncode is None:
                    proc.terminate()

                port_arg = data.get("port", "")
                confidence = data.get("confidence", "0.5")

                pen_cmd = [sys.executable, PENNFERENCE_PATH, "--confidence", str(confidence)]
                if port_arg:
                    pen_cmd.extend(["--port", port_arg])

                print(f"[pen] starting: {' '.join(pen_cmd)}")
                proc = await asyncio.create_subprocess_exec(
                    *pen_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=PROJECT_ROOT,
                )

                await ws_resp.send_str(json.dumps({"type": "pen_started"}))

                # Read stdout line by line and parse recognized words
                async def read_output():
                    try:
                        while True:
                            line = await proc.stdout.readline()
                            if not line:
                                break
                            text = line.decode().strip()
                            print(f"[pen] {text}")

                            # pennference.py outputs "  >>> word1 word2 word3"
                            # when it has accumulated words
                            if text.startswith(">>>"):
                                all_words = text[3:].strip()
                                # Send only the latest word (last one in the list)
                                words = all_words.split()
                                if words:
                                    latest = words[-1]
                                    if not ws_resp.closed:
                                        await ws_resp.send_str(json.dumps({
                                            "type": "word",
                                            "word": latest,
                                            "total": len(words),
                                        }))

                            # Also detect writing state changes
                            elif "WORD STARTED" in text:
                                if not ws_resp.closed:
                                    await ws_resp.send_str(json.dumps({"type": "writing"}))

                            elif "WORD IS:" in text:
                                # "  HH:MM:SS  WORD IS: hello  (85%)"
                                parts = text.split("WORD IS:")
                                if len(parts) > 1:
                                    word_conf = parts[1].strip()
                                    word = word_conf.split("(")[0].strip()
                                    conf = ""
                                    if "(" in word_conf:
                                        conf = word_conf.split("(")[1].rstrip(")")
                                    if not ws_resp.closed:
                                        await ws_resp.send_str(json.dumps({
                                            "type": "word",
                                            "word": word,
                                            "confidence": conf,
                                        }))

                            elif "LOW CONFIDENCE" in text:
                                parts = text.split("LOW CONFIDENCE:")
                                if len(parts) > 1:
                                    word_conf = parts[1].strip().split("--")[0].strip()
                                    word = word_conf.split("(")[0].strip()
                                    if not ws_resp.closed:
                                        await ws_resp.send_str(json.dumps({
                                            "type": "low_confidence",
                                            "word": word,
                                        }))

                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        print(f"[pen] read error: {e}")
                    finally:
                        if not ws_resp.closed:
                            await ws_resp.send_str(json.dumps({"type": "pen_stopped"}))

                asyncio.create_task(read_output())

                # Also read stderr for connection errors
                async def read_errors():
                    try:
                        while True:
                            line = await proc.stderr.readline()
                            if not line:
                                break
                            err = line.decode().strip()
                            if err:
                                print(f"[pen] stderr: {err}")
                    except Exception:
                        pass

                asyncio.create_task(read_errors())

            elif cmd == "stop":
                if proc and proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=3)
                    except asyncio.TimeoutError:
                        proc.kill()
                    proc = None
                await ws_resp.send_str(json.dumps({"type": "pen_stopped"}))

    except Exception as e:
        print(f"[pen] ws error: {e}")
    finally:
        if proc and proc.returncode is None:
            proc.terminate()

    return ws_resp


app = web.Application()
app.router.add_get("/", notes_page)
app.router.add_get("/static/{filename}", static_handler)
app.router.add_post("/api/ara", api_ara)
app.router.add_get("/ws/pen", ws_pen)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8767"))
    print(f"ThePenn Notes UI: http://localhost:{port}")
    print(f"Ara app: {ARA_APP_PATH}")
    print(f"Pennference: {PENNFERENCE_PATH}")
    print()
    web.run_app(app, host="localhost", port=port, print=None)
