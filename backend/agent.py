"""
agent.py — Agentic decision engine.

Watches gesture changes, queries Groq (with LRU cache), falls back to
a hardcoded rule table, then calls execute_action().
"""

import logging
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Tuple

from actions import VALID_ACTIONS, execute_action
from gesture import CONFIDENCE_THRESHOLD

if TYPE_CHECKING:
    from main import AppState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App detection & normalisation
# ---------------------------------------------------------------------------

APP_CATEGORIES = {
    "browser":  ["chrome", "firefox", "edge", "safari", "brave", "opera"],
    "slides":   ["powerpoint", "impress", "keynote", "slides", "presentation"],
    "media":    ["vlc", "spotify", "mpv", "wmplayer", "music", "video"],
    "terminal": ["terminal", "cmd", "powershell", "bash", "wt", "windows terminal"],
    "code":     ["code", "vscode", "pycharm", "intellij", "vim", "nvim", "sublime"],
}


def _normalize_app(raw_title: str) -> str:
    lower = raw_title.lower()
    for category, keywords in APP_CATEGORIES.items():
        if any(kw in lower for kw in keywords):
            return category
    return "other"


def _get_raw_window_title() -> str:
    if sys.platform == "win32":
        try:
            import pygetwindow as gw
            win = gw.getActiveWindow()
            return win.title if win else ""
        except Exception:
            return ""
    elif sys.platform == "darwin":
        try:
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get name of first process whose frontmost is true',
                ],
                capture_output=True, text=True, timeout=1,
            )
            return result.stdout.strip()
        except Exception:
            return ""
    else:  # Linux
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=1,
            )
            return result.stdout.strip()
        except Exception:
            return ""


def get_active_app() -> str:
    """Return normalised app category for the currently focused window."""
    raw = _get_raw_window_title()
    return _normalize_app(raw)


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------

class LRUCache:
    def __init__(self, maxsize: int = 10):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: tuple) -> Optional[str]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Fallback rule table
# ---------------------------------------------------------------------------

_FALLBACK_RULES: dict = {
    ("PINCH",        "any"):     "volume_adjust",
    ("SWIPE_RIGHT",  "browser"): "next_tab",
    ("SWIPE_RIGHT",  "slides"):  "next_slide",
    ("SWIPE_RIGHT",  "media"):   "next_track",
    ("SWIPE_LEFT",   "browser"): "prev_tab",
    ("SWIPE_LEFT",   "slides"):  "prev_slide",
    ("SWIPE_LEFT",   "media"):   "prev_track",
    ("OPEN_PALM",    "any"):     "pause_play",
    ("INDEX_UP",     "any"):     "move_cursor",
    ("FIST",         "any"):     "left_click",
}

_MODE_OVERRIDES: dict = {
    "volume":  {"SWIPE_RIGHT": "volume_up",  "SWIPE_LEFT": "volume_down",
                "PINCH": "volume_adjust"},
    "slides":  {"SWIPE_RIGHT": "next_slide", "SWIPE_LEFT": "prev_slide",
                "OPEN_PALM": "pause_play"},
    "cursor":  {"INDEX_UP": "move_cursor",   "FIST": "left_click"},
}


def _fallback_action(gesture: str, app: str, mode: str) -> str:
    if mode in _MODE_OVERRIDES:
        action = _MODE_OVERRIDES[mode].get(gesture)
        if action:
            return action

    action = _FALLBACK_RULES.get((gesture, app)) or _FALLBACK_RULES.get((gesture, "any"))
    return action or "none"


# ---------------------------------------------------------------------------
# Groq integration
# ---------------------------------------------------------------------------

_GROQ_SYSTEM = (
    "Map gesture+app to action. Reply ONE word only. "
    "Actions: volume_up,volume_down,volume_adjust,next_tab,prev_tab,"
    "next_slide,prev_slide,next_track,prev_track,pause_play,"
    "move_cursor,left_click,scroll_up,scroll_down,none"
)


def _build_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — Groq disabled, using fallback rules.")
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        logger.warning("groq package not installed — using fallback rules.")
        return None


def _call_groq(client, gesture: str, app: str, mode: str) -> Optional[str]:
    """Send a compact prompt to Groq. Returns a valid action string or None."""
    if client is None:
        return None

    # User message: well under 30 tokens
    user_msg = f"gesture:{gesture} app:{app} mode:{mode} -> action?"

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": _GROQ_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=5,
            temperature=0.0,
            timeout=2.0,
        )
        raw = response.choices[0].message.content.strip().lower()
        # Strip punctuation / extra words
        action = raw.split()[0] if raw else ""
        if action in VALID_ACTIONS:
            return action
        logger.debug(f"Groq returned unknown action '{raw}', falling back.")
        return None
    except Exception as e:
        logger.warning(f"Groq call failed ({type(e).__name__}): {e} — using fallback.")
        return None


# ---------------------------------------------------------------------------
# Agent engine
# ---------------------------------------------------------------------------

class AgentEngine:
    GESTURE_DEBOUNCE = 2.0  # seconds — don't re-call for unchanged gesture

    def __init__(self, state: "AppState"):
        self.state = state
        self._cache = LRUCache(maxsize=10)
        self._groq_client = _build_groq_client()
        self._last_gesture_sent = ""
        self._last_sent_time: float = 0.0
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="AgentThread")
        self._thread.start()

    def _process_gesture(
        self, gesture: str, app: str, mode: str, state: "AppState"
    ) -> Optional[str]:
        """
        Core decision logic: cache → Groq → fallback → execute.
        Extracted so tests can call it directly without running the loop thread.
        Returns the action taken (or "none").
        """
        cache_key = (gesture, app, mode)
        action = self._cache.get(cache_key)

        if action is None:
            action = _call_groq(self._groq_client, gesture, app, mode)
            if action is None:
                action = _fallback_action(gesture, app, mode)
            self._cache.put(cache_key, action)
        else:
            logger.debug(f"Cache hit: {cache_key} → {action}")

        if action and action != "none":
            execute_action(action, state)
            log_entry = f"{time.strftime('%H:%M:%S')} | {gesture} + {app} [{mode}] → {action}"
            with state.lock:
                state.last_action = action
                state.action_log.appendleft(log_entry)
            logger.info(log_entry)

        return action or "none"

    def _loop(self) -> None:
        while not self.state.stop_event.is_set():
            time.sleep(0.1)  # 10 Hz poll

            with self.state.lock:
                gesture = self.state.gesture.name
                confidence = self.state.gesture.confidence
                mode = self.state.current_mode

            # Skip low-confidence or missing gestures
            if gesture == "NONE" or confidence < CONFIDENCE_THRESHOLD:
                continue

            now = time.monotonic()

            # Skip if gesture is unchanged (debounce)
            if gesture == self._last_gesture_sent:
                continue

            # New gesture detected
            self._last_gesture_sent = gesture
            self._last_sent_time = now

            # Detect active app
            app = get_active_app()
            with self.state.lock:
                self.state.active_app = app

            self._process_gesture(gesture, app, mode, self.state)
