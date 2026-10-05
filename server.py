#!/usr/bin/env python3
"""JARVIS Local: voice assistant that drives Claude Code sessions.

One small FastAPI server:
  GET  /              the futuristic UI (orb + live task panels)
  POST /api/transcribe  transcribes audio via Ollama (Whisper) - free
  POST /api/chat        generates a response via Ollama (llama3) - free
  POST /api/task      spawns a Claude Code session (claude -p) in background
  GET  /api/task/{id} polls a task's status and output

The browser uses the Web Speech API for voice recognition (free), and the
server uses Ollama for transcription and chat generation. The model calls
the delegate_to_claude tool for any real work.
"""
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent
app = FastAPI(title="JARVIS Local")

# ---------------------------------------------------------------- config

def load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
REALTIME_MODEL = os.environ.get("REALTIME_MODEL", "gpt-realtime")
VOICE = os.environ.get("JARVIS_VOICE", "ballad")
LANGUAGE = os.environ.get("JARVIS_LANGUAGE", "français")
# Where Claude Code sessions run (their filesystem playground).
WORKDIR = os.path.expanduser(os.environ.get("JARVIS_WORKDIR", "~"))

# Headless sessions have nobody to answer permission prompts: a task that asks
# would just hang until the timeout. Run them in a non-interactive mode instead.
# bypassPermissions = no prompt at all; acceptEdits = files yes, commands still ask.
PERMISSION_MODE = os.environ.get("JARVIS_PERMISSION_MODE", "bypassPermissions")

INSTRUCTIONS = f"""Tu es JARVIS, l'assistant vocal personnel de monsieur, dans
l'esprit du majordome d'Iron Man. Tu parles en {LANGUAGE} avec un LÉGER ACCENT
BRITANNIQUE distingué et un flegme impeccable: voix posée, articulation
soignée, débit calme, jamais d'exubérance. Tu t'adresses à l'utilisateur par
"monsieur", avec une courtoisie raffinée et une pointe d'esprit pince-sans-rire
("Très bien, monsieur.", "Si monsieur veut bien patienter un instant.").
Réponses COURTES (une ou deux phrases), naturelles et directes.

Pour toute tâche réelle (lire ou créer des fichiers, chercher sur internet,
coder, analyser, automatiser), tu appelles l'outil delegate_to_claude avec un
prompt clair et complet. Tu annonces brièvement que tu lances la tâche, puis
tu continues la conversation. Quand un résultat de tâche arrive, tu le
résumes à voix haute en une ou deux phrases.

Pour ouvrir un logiciel sur ce PC ("lance Discord", "ouvre Spotify"), tu
appelles open_app avec le nom de l'application. Pour un site web ou service
en ligne ("ouvre mes emails" -> https://mail.google.com, "ouvre YouTube"),
appelle open_url avec l'URL complète. Si l'utilisateur précise un écran
("sur l'écran de gauche", "à droite", "sur l'écran 2"), passe monitor
(left/right/top/bottom/primary ou un numéro). Tu peux enchaîner plusieurs
appels pour installer un setup multi-écrans. Confirme brièvement.

Si l'utilisateur demande d'annuler ou d'arrêter une tâche en cours, appelle
cancel_task (sans task_id pour la plus récente).

Quand tu veux MONTRER quelque chose à l'écran (résultat de calcul, liste,
tableau, extrait de code, définition), appelle display_card: le contenu
s'affiche sur l'interface. Utilise-la spontanément dès qu'un visuel aide
(chiffres, comparaisons, étapes), et garde ta réponse vocale courte.

Pour une ANALYSE DE DONNÉES ou un rapport (fichier Excel/CSV analysé, stats,
comparatifs chiffrés), appelle display_report: un tableau de bord s'affiche
avec indicateurs clés (kpis), graphique (chart) et tableau (table). Quand tu
délègues une analyse à delegate_to_claude, demande-lui explicitement de
terminer sa réponse par les données chiffrées structurées (listes de valeurs,
totaux, moyennes) pour que tu puisses remplir le rapport ensuite.

Ne réponds jamais de mémoire à une question qui demande des données réelles:
délègue. Ne lis jamais de longues listes: résume."""

TOOLS = [{
    "type": "function",
    "name": "delegate_to_claude",
    "description": ("Delegate a real task to a Claude Code session running on "
                    "this machine (files, code, web research, automation). "
                    "Returns immediately; the result arrives later as a "
                    "system message."),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Very short task label (3-5 words)"},
            "prompt": {"type": "string", "description": "Complete, self-contained task instruction for Claude Code"},
        },
        "required": ["title", "prompt"],
    },
}, {
    "type": "function",
    "name": "open_app",
    "description": ("Launch an application installed on this PC by name "
                    "(e.g. 'discord', 'spotify', 'chrome', 'notepad'). "
                    "Returns whether it was found and started."),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Application name as the user said it"},
            "monitor": {"type": "string",
                        "description": ("Target screen: 'left', 'right', 'top', "
                                        "'bottom', 'primary', or a number like '2'. "
                                        "Omit to leave window placement alone.")},
        },
        "required": ["name"],
    },
}, {
    "type": "function",
    "name": "open_url",
    "description": ("Open a website in the browser on this PC. Use for online "
                    "services: 'mes emails' -> https://mail.google.com, "
                    "'YouTube' -> https://youtube.com, etc."),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL to open (https://...)"},
            "monitor": {"type": "string",
                        "description": "Target screen: 'left', 'right', 'top', 'bottom', 'primary' or a number. Optional."},
        },
        "required": ["url"],
    },
}, {
    "type": "function",
    "name": "cancel_task",
    "description": ("Cancel a running Claude Code task. Omit task_id to cancel "
                    "the most recently started running task."),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Task id to cancel (optional)"},
        },
    },
}, {
    "type": "function",
    "name": "display_card",
    "description": ("Show a visual card on the JARVIS screen: results, "
                    "numbers, lists, code, comparisons. Use markdown-lite: "
                    "**bold**, `code`, lines starting with '- ' for bullets. "
                    "Use whenever a visual helps; keep the spoken reply short."),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short card title"},
            "content": {"type": "string", "description": "Card body (markdown-lite)"},
            "kind": {"type": "string", "enum": ["info", "result", "code", "warning"],
                      "description": "Visual style of the card"},
        },
        "required": ["title", "content"],
    },
}, {
    "type": "function",
    "name": "display_report",
    "description": ("Show a full data report dashboard on screen: KPI tiles, "
                    "an interactive chart, a sortable table, and markdown notes. "
                    "Use for data analysis results (spreadsheets, stats, "
                    "comparisons). All sections are optional except title."),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Report title"},
            "kpis": {"type": "array", "description": "Headline numbers (max 4)",
                     "items": {"type": "object", "properties": {
                         "label": {"type": "string"},
                         "value": {"type": "string", "description": "e.g. '12 480 €'"},
                         "delta": {"type": "string", "description": "e.g. '+12%' (optional)"},
                     }, "required": ["label", "value"]}},
            "chart": {"type": "object", "description": "One chart", "properties": {
                "type": {"type": "string", "enum": ["line", "bar", "area", "donut"]},
                "categories": {"type": "array", "items": {"type": "string"},
                               "description": "X axis labels (or slice labels for donut)"},
                "series": {"type": "array", "description": "1-3 series",
                           "items": {"type": "object", "properties": {
                               "name": {"type": "string"},
                               "data": {"type": "array", "items": {"type": "number"}},
                           }, "required": ["name", "data"]}},
            }},
            "table": {"type": "object", "properties": {
                "columns": {"type": "array", "items": {"type": "string"}},
                "rows": {"type": "array", "items": {"type": "array",
                         "items": {"type": ["string", "number"]}}},
            }},
            "markdown": {"type": "string", "description": "Notes / conclusions in markdown"},
        },
        "required": ["title"],
    },
}]

# ---------------------------------------------------------------- tasks

TASKS: dict = {}
PROCS: dict = {}  # task_id -> Popen, kept out of TASKS so get_task stays JSON-safe


def _kill_tree(pid: int):
    """Kill a process and its children (claude.cmd spawns node)."""
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)


def _run_task(task_id: str, prompt: str):
    task = TASKS[task_id]
    try:
        # shutil.which honours PATHEXT, so this also finds claude.cmd on Windows.
        claude = shutil.which("claude")
        if not claude:
            raise FileNotFoundError("claude")
        cmd = [claude, "-p", prompt]
        if PERMISSION_MODE and PERMISSION_MODE.lower() != "off":
            cmd += ["--permission-mode", PERMISSION_MODE]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", cwd=WORKDIR,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        PROCS[task_id] = proc
        try:
            stdout, stderr = proc.communicate(
                timeout=int(os.environ.get("JARVIS_TASK_TIMEOUT", "600")))
        except subprocess.TimeoutExpired:
            _kill_tree(proc.pid)
            proc.communicate()
            task["status"] = "error"
            task["output"] = "Timeout: la session Claude a dépassé la limite de temps."
        else:
            if task["status"] == "cancelled":
                pass  # set by the cancel endpoint; don't overwrite
            else:
                out = (stdout or "").strip()
                err = (stderr or "").strip()
                task["status"] = "done" if proc.returncode == 0 else "error"
                task["output"] = out if out else err[:2000]
    except FileNotFoundError:
        task["status"] = "error"
        task["output"] = ("La commande 'claude' est introuvable. Installe Claude Code: "
                          "npm install -g @anthropic-ai/claude-code")
    except Exception as exc:  # noqa: BLE001
        task["status"] = "error"
        task["output"] = str(exc)
    finally:
        PROCS.pop(task_id, None)
    task["ended"] = time.time()


class TaskIn(BaseModel):
    title: str
    prompt: str


@app.post("/api/task")
def create_task(body: TaskIn):
    task_id = uuid.uuid4().hex[:8]
    TASKS[task_id] = {
        "id": task_id, "title": body.title, "prompt": body.prompt,
        "status": "running", "output": "", "started": time.time(), "ended": None,
    }
    threading.Thread(target=_run_task, args=(task_id, body.prompt), daemon=True).start()
    return {"id": task_id, "status": "running"}


@app.post("/api/task/{task_id}/cancel")
def cancel_task(task_id: str):
    if task_id in ("latest", "last", "-"):
        running = [t for t in TASKS.values() if t["status"] == "running"]
        if not running:
            return {"ok": False, "error": "Aucune tâche en cours."}
        task = max(running, key=lambda t: t["started"])
    else:
        task = TASKS.get(task_id)
        if not task:
            raise HTTPException(404, "unknown task")
        if task["status"] != "running":
            return {"ok": False, "error": f"La tâche est déjà {task['status']}."}
    # Flag first so _run_task's communicate() return doesn't overwrite it.
    task["status"] = "cancelled"
    task["output"] = "Annulée par l'utilisateur."
    proc = PROCS.get(task["id"])
    if proc and proc.poll() is None:
        _kill_tree(proc.pid)
    return {"ok": True, "cancelled": task["id"], "title": task["title"]}


@app.get("/api/task/{task_id}")
def get_task(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(404, "unknown task")
    return task


# ---------------------------------------------------------------- monitors & window placement (Windows API)

import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
try:  # accurate multi-monitor coordinates under display scaling
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:  # noqa: BLE001
    pass

_MonitorEnumProc = ctypes.WINFUNCTYPE(
    ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
    ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
_EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HWND, wintypes.LPARAM)


def _monitors():
    """List monitor work rects as (left, top, right, bottom)."""
    mons = []

    def cb(hmon, hdc, lprc, lparam):
        r = lprc.contents
        mons.append((r.left, r.top, r.right, r.bottom))
        return 1

    user32.EnumDisplayMonitors(0, 0, _MonitorEnumProc(cb), 0)
    return mons


def _pick_monitor(target: str):
    mons = _monitors()
    if not mons:
        return None
    t = (target or "").strip().lower()
    if t.isdigit():
        i = int(t) - 1
        return mons[i] if 0 <= i < len(mons) else None
    key = {
        "left": lambda m: m[0], "gauche": lambda m: m[0],
        "top": lambda m: m[1], "haut": lambda m: m[1],
    }
    if t in key:
        return min(mons, key=key[t])
    key = {
        "right": lambda m: m[2], "droite": lambda m: m[2], "droit": lambda m: m[2],
        "bottom": lambda m: m[3], "bas": lambda m: m[3],
    }
    if t in key:
        return max(mons, key=key[t])
    # primary: the monitor containing the origin (0,0)
    for m in mons:
        if m[0] <= 0 < m[2] and m[1] <= 0 < m[3]:
            return m
    return mons[0]


def _visible_windows():
    """Map of visible top-level windows: hwnd -> title."""
    wins = {}

    def cb(hwnd, lparam):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                wins[hwnd] = buf.value
        return 1

    user32.EnumWindows(_EnumWindowsProc(cb), 0)
    return wins


def _move_to_monitor(hwnd, mon):
    left, top, right, bottom = mon
    SW_RESTORE, SW_MAXIMIZE = 9, 3
    user32.ShowWindow(hwnd, SW_RESTORE)  # a maximized window can't be moved
    user32.MoveWindow(hwnd, left + 40, top + 40,
                      max(400, (right - left) - 80), max(300, (bottom - top) - 80), True)
    user32.ShowWindow(hwnd, SW_MAXIMIZE)
    user32.SetForegroundWindow(hwnd)


def _place_app_window(app_name: str, before: dict, mon, timeout: float = 20.0):
    """Wait for the app's window to appear, then move it to the target monitor.

    Prefers a NEW window whose title mentions the app; falls back to any new
    window, then to an existing title match (single-instance apps like Discord
    just refocus their already-open window).
    """
    q = app_name.lower()
    deadline = time.time() + timeout
    fallback = None
    while time.time() < deadline:
        wins = _visible_windows()
        new = {h: t for h, t in wins.items() if h not in before}
        for h, title in new.items():
            if q in title.lower():
                _move_to_monitor(h, mon)
                return title
        if new and fallback is None:
            fallback = max(new)  # remember, but keep hoping for a title match
        time.sleep(0.5)
        if fallback and time.time() > deadline - timeout / 2:
            break
    if fallback:
        wins = _visible_windows()
        _move_to_monitor(fallback, mon)
        return wins.get(fallback, app_name)
    # No new window: single-instance app already running -> match existing title.
    for h, title in _visible_windows().items():
        if q in title.lower():
            _move_to_monitor(h, mon)
            return title
    return None


# ---------------------------------------------------------------- open app

START_MENU_DIRS = [
    Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
    Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
]


def _find_shortcut(name: str):
    """Fuzzy-match a Start Menu shortcut (where installed apps register)."""
    q = name.lower().strip()
    best, best_score = None, 0.0
    for root in START_MENU_DIRS:
        if not root.is_dir():
            continue
        for lnk in root.rglob("*.lnk"):
            stem = lnk.stem.lower()
            if q == stem:
                return lnk
            score = 0.0
            if q in stem:
                score = 2 + len(q) / len(stem)   # substring: prefer tightest match
            elif all(w in stem for w in q.split()):
                score = 1
            # Penalise uninstallers and docs.
            if any(bad in stem for bad in ("uninstall", "désinstaller", "readme", "website")):
                score -= 2
            if score > best_score:
                best, best_score = lnk, score
    return best


class OpenIn(BaseModel):
    name: str = ""
    url: str | None = None
    monitor: str | None = None


def _placed(name: str, monitor: str | None, before: dict, launched: str):
    """Optionally move the freshly launched app to the requested screen."""
    if not monitor:
        return {"ok": True, "launched": launched}
    mon = _pick_monitor(monitor)
    if not mon:
        return {"ok": True, "launched": launched,
                "warning": f"écran '{monitor}' introuvable, fenêtre laissée en place"}
    title = _place_app_window(name, before, mon)
    if title:
        return {"ok": True, "launched": launched, "monitor": monitor, "window": title}
    return {"ok": True, "launched": launched,
            "warning": "fenêtre non détectée, placement impossible"}


@app.post("/api/open")
def open_app(body: OpenIn):
    name = body.name.strip()
    if body.url:
        url = body.url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        before = _visible_windows() if body.monitor else {}
        import webbrowser
        if not webbrowser.open(url):
            return {"ok": False, "error": "Impossible d'ouvrir le navigateur."}
        # Match the browser window by the site's domain.
        domain = url.split("//", 1)[1].split("/", 1)[0].removeprefix("www.")
        return _placed(domain.split(".")[0], body.monitor, before, url)
    if not name:
        raise HTTPException(400, "missing app name or url")
    before = _visible_windows() if body.monitor else {}
    lnk = _find_shortcut(name)
    if lnk:
        os.startfile(lnk)  # noqa: S606 - deliberate: local launcher
        return _placed(name, body.monitor, before, lnk.stem)
    # Fallback: resolve via PATH then the App Paths registry (chrome, notepad...).
    exe = shutil.which(name) or shutil.which(name + ".exe")
    if not exe:
        try:
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    key = winreg.OpenKey(hive, rf"Software\Microsoft\Windows"
                                               rf"\CurrentVersion\App Paths\{name}.exe")
                    exe = winreg.QueryValueEx(key, None)[0].strip('"')
                    break
                except OSError:
                    continue
        except ImportError:
            pass
    if not exe:
        return {"ok": False, "error": f"Application '{name}' introuvable sur ce PC."}
    try:
        subprocess.Popen([exe], cwd=str(Path(exe).parent))
        res = _placed(name, body.monitor, before, Path(exe).stem)
        res.setdefault("via", "exe")
        return res
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------- Ollama routes (free, local)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

@app.post("/api/transcribe")
async def transcribe(request: Request):
    """Transcribe audio using Ollama (Whisper) - free, no API key needed."""
    try:
        body = await request.body()
        # Save audio to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(body)
            audio_path = f.name
        
        # Use Ollama to transcribe
        r = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "whisper",
                "prompt": f"Transcribe this audio file: {audio_path}",
            },
            timeout=120,
        )
        os.unlink(audio_path)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, f"Ollama: {r.text[:300]}")
        data = r.json()
        return {"text": data.get("response", "")}
    except Exception as exc:
        raise HTTPException(500, str(exc))

@app.post("/api/chat")
async def chat(request: Request):
    """Generate a response using Ollama (llama3) - free, no API key needed."""
    try:
        body = await request.json()
        messages = body.get("messages", [])
        
        # Format messages for Ollama
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"[SYSTEM] {content}\n"
            elif role == "user":
                prompt += f"[USER] {content}\n"
            elif role == "assistant":
                prompt += f"[ASSISTANT] {content}\n"
        prompt += "[ASSISTANT] "
        
        r = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        if r.status_code >= 400:
            raise HTTPException(r.status_code, f"Ollama: {r.text[:300]}")
        data = r.json()
        return {"response": data.get("response", "")}
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ---------------------------------------------------------------- static

@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("JARVIS_PORT", "8788"))
    print(f"\n  JARVIS Local -> http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
