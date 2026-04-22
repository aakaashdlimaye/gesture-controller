"""
actions.py — OS-level action execution layer.

Wraps every action in try/except and logs failures.
Cross-platform: Windows (pycaw + pyautogui), Linux (pactl + pyautogui), Mac (osascript + pyautogui).
"""

import logging
import subprocess
import sys
from typing import TYPE_CHECKING, Tuple

import pyautogui

if TYPE_CHECKING:
    from main import AppState

logger = logging.getLogger(__name__)

# Disable pyautogui fail-safe (moving mouse to corner would raise an exception)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # No automatic delay between pyautogui calls

# ---------------------------------------------------------------------------
# Volume control
# ---------------------------------------------------------------------------

def _volume_win(delta: float) -> None:
    """Adjust Windows volume by `delta` (e.g. +0.05 or -0.05)."""
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        from comtypes import CLSCTX_ALL
    except ImportError:
        logger.error("pycaw not installed; cannot adjust Windows volume.")
        return

    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = interface.QueryInterface(IAudioEndpointVolume)
        current = volume.GetMasterVolumeLevelScalar()
        new_vol = max(0.0, min(1.0, current + delta))
        volume.SetMasterVolumeLevelScalar(new_vol, None)
    except Exception as e:
        logger.error(f"_volume_win failed: {e}")


def _volume_linux(delta_pct: int) -> None:
    """Adjust Linux/PulseAudio volume by ±N%."""
    try:
        sign = "+" if delta_pct >= 0 else "-"
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{sign}{abs(delta_pct)}%"],
            timeout=2,
            check=True,
        )
    except Exception as e:
        logger.error(f"_volume_linux failed: {e}")


def _volume_mac(delta_pct: int) -> None:
    """Adjust macOS volume by ±N%."""
    try:
        # Get current volume (0-100)
        result = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True, timeout=2,
        )
        current = int(result.stdout.strip())
        new_vol = max(0, min(100, current + delta_pct))
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {new_vol}"],
            timeout=2, check=True,
        )
    except Exception as e:
        logger.error(f"_volume_mac failed: {e}")


def action_volume_up() -> None:
    try:
        if sys.platform == "win32":
            _volume_win(+0.05)
        elif sys.platform == "darwin":
            _volume_mac(+5)
        else:
            _volume_linux(+5)
    except Exception as e:
        logger.error(f"action_volume_up failed: {e}")


def action_volume_down() -> None:
    try:
        if sys.platform == "win32":
            _volume_win(-0.05)
        elif sys.platform == "darwin":
            _volume_mac(-5)
        else:
            _volume_linux(-5)
    except Exception as e:
        logger.error(f"action_volume_down failed: {e}")


def action_volume_adjust(state: "AppState") -> None:
    """PINCH gesture — use pinch confidence to decide up/down."""
    try:
        with state.lock:
            conf = state.gesture.confidence
        # confidence near 1.0 means tight pinch → volume down; near 0.75 → up
        if conf > 0.88:
            action_volume_down()
        else:
            action_volume_up()
    except Exception as e:
        logger.error(f"action_volume_adjust failed: {e}")


# ---------------------------------------------------------------------------
# Keyboard / media actions
# ---------------------------------------------------------------------------

def action_next_tab() -> None:
    try:
        pyautogui.hotkey("ctrl", "tab")
    except Exception as e:
        logger.error(f"action_next_tab failed: {e}")


def action_prev_tab() -> None:
    try:
        pyautogui.hotkey("ctrl", "shift", "tab")
    except Exception as e:
        logger.error(f"action_prev_tab failed: {e}")


def action_next_slide() -> None:
    try:
        pyautogui.press("right")
    except Exception as e:
        logger.error(f"action_next_slide failed: {e}")


def action_prev_slide() -> None:
    try:
        pyautogui.press("left")
    except Exception as e:
        logger.error(f"action_prev_slide failed: {e}")


def action_next_track() -> None:
    try:
        pyautogui.hotkey("ctrl", "right")
    except Exception as e:
        logger.error(f"action_next_track failed: {e}")


def action_prev_track() -> None:
    try:
        pyautogui.hotkey("ctrl", "left")
    except Exception as e:
        logger.error(f"action_prev_track failed: {e}")


def action_pause_play() -> None:
    try:
        pyautogui.press("space")
    except Exception as e:
        logger.error(f"action_pause_play failed: {e}")


def action_left_click() -> None:
    try:
        pyautogui.click()
    except Exception as e:
        logger.error(f"action_left_click failed: {e}")


def action_scroll_up() -> None:
    try:
        pyautogui.scroll(3)
    except Exception as e:
        logger.error(f"action_scroll_up failed: {e}")


def action_scroll_down() -> None:
    try:
        pyautogui.scroll(-3)
    except Exception as e:
        logger.error(f"action_scroll_down failed: {e}")


# ---------------------------------------------------------------------------
# Cursor control
# ---------------------------------------------------------------------------

def action_move_cursor(lm_x: float, lm_y: float) -> None:
    """
    Move the cursor to the position indicated by the INDEX_UP fingertip.
    lm_x, lm_y are normalised MediaPipe coordinates [0, 1].
    Mirrors x-axis because the webcam image is flipped relative to the user.
    """
    try:
        screen_w, screen_h = pyautogui.size()
        target_x = int((1.0 - lm_x) * screen_w)
        target_y = int(lm_y * screen_h)
        pyautogui.moveTo(target_x, target_y, duration=0.05)
    except Exception as e:
        logger.error(f"action_move_cursor failed: {e}")


# ---------------------------------------------------------------------------
# Central dispatcher
# ---------------------------------------------------------------------------

VALID_ACTIONS = {
    "volume_up", "volume_down", "volume_adjust",
    "next_tab", "prev_tab",
    "next_slide", "prev_slide",
    "next_track", "prev_track",
    "pause_play",
    "left_click",
    "move_cursor",
    "scroll_up", "scroll_down",
    "none",
}


def execute_action(action: str, state: "AppState") -> None:
    """Dispatch an action string to the appropriate function."""
    if action == "none" or action not in VALID_ACTIONS:
        return

    dispatch = {
        "volume_up":     action_volume_up,
        "volume_down":   action_volume_down,
        "volume_adjust": lambda: action_volume_adjust(state),
        "next_tab":      action_next_tab,
        "prev_tab":      action_prev_tab,
        "next_slide":    action_next_slide,
        "prev_slide":    action_prev_slide,
        "next_track":    action_next_track,
        "prev_track":    action_prev_track,
        "pause_play":    action_pause_play,
        "left_click":    action_left_click,
        "scroll_up":     action_scroll_up,
        "scroll_down":   action_scroll_down,
        "move_cursor": lambda: (
            action_move_cursor(*state.cursor_target)
            if state.cursor_target != (0.0, 0.0)
            else None
        ),
    }

    handler = dispatch.get(action)
    if handler:
        try:
            handler()
        except Exception as e:
            logger.error(f"execute_action('{action}') failed: {e}")
