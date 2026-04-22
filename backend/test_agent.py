"""
test_agent.py — Unit tests for the agentic decision engine.

Tests: LRU cache, Groq call path, cache hits, fallback rules,
debounce, and timeout fallback.
"""

import sys
import threading
import time
import unittest
from collections import deque
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal AppState stub (avoids importing the full FastAPI main.py)
# ---------------------------------------------------------------------------

@dataclass
class _GestureState:
    name: str = "NONE"
    confidence: float = 0.0
    timestamp: float = 0.0


@dataclass
class _AppState:
    gesture: _GestureState = field(default_factory=_GestureState)
    fps: float = 0.0
    webcam_active: bool = False
    cursor_target: tuple = field(default_factory=lambda: (0.0, 0.0))
    active_app: str = "unknown"
    last_action: str = "none"
    current_mode: str = "auto"
    action_log: deque = field(default_factory=lambda: deque(maxlen=50))
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


# Inject stub into sys.modules so agent.py can do `from main import AppState`
import types
fake_main = types.ModuleType("main")
fake_main.AppState = _AppState
fake_main.GestureState = _GestureState
sys.modules.setdefault("main", fake_main)

# Stub out pyautogui to prevent real keypresses during tests
pyautogui_mock = MagicMock()
sys.modules.setdefault("pyautogui", pyautogui_mock)

from agent import (
    LRUCache,
    _call_groq,
    _fallback_action,
    _normalize_app,
    AgentEngine,
)


# ---------------------------------------------------------------------------
# LRU Cache tests
# ---------------------------------------------------------------------------

class TestLRUCache(unittest.TestCase):

    def test_miss_returns_none(self):
        cache = LRUCache(maxsize=3)
        self.assertIsNone(cache.get(("PINCH", "browser", "auto")))

    def test_put_and_get(self):
        cache = LRUCache(maxsize=3)
        cache.put(("PINCH", "browser", "auto"), "volume_adjust")
        self.assertEqual(cache.get(("PINCH", "browser", "auto")), "volume_adjust")

    def test_eviction_at_capacity(self):
        cache = LRUCache(maxsize=3)
        cache.put(("A", "x", "auto"), "a1")
        cache.put(("B", "x", "auto"), "b1")
        cache.put(("C", "x", "auto"), "c1")
        # Adding 4th should evict oldest ("A")
        cache.put(("D", "x", "auto"), "d1")
        self.assertEqual(len(cache), 3)
        self.assertIsNone(cache.get(("A", "x", "auto")))
        self.assertEqual(cache.get(("D", "x", "auto")), "d1")

    def test_access_refreshes_recency(self):
        cache = LRUCache(maxsize=3)
        cache.put(("A", "x", "auto"), "a1")
        cache.put(("B", "x", "auto"), "b1")
        cache.put(("C", "x", "auto"), "c1")
        # Access A → it becomes most recent
        cache.get(("A", "x", "auto"))
        # Add D → should evict B (now oldest), not A
        cache.put(("D", "x", "auto"), "d1")
        self.assertIsNotNone(cache.get(("A", "x", "auto")))
        self.assertIsNone(cache.get(("B", "x", "auto")))

    def test_capacity_10(self):
        cache = LRUCache(maxsize=10)
        for i in range(11):
            cache.put((str(i), "x", "auto"), f"action{i}")
        self.assertEqual(len(cache), 10)


# ---------------------------------------------------------------------------
# App normalisation tests
# ---------------------------------------------------------------------------

class TestNormalizeApp(unittest.TestCase):

    def test_chrome_is_browser(self):
        self.assertEqual(_normalize_app("Google Chrome - New Tab"), "browser")

    def test_powerpoint_is_slides(self):
        self.assertEqual(_normalize_app("Quarterly Review.pptx - PowerPoint"), "slides")

    def test_vlc_is_media(self):
        self.assertEqual(_normalize_app("VLC media player"), "media")

    def test_unknown_is_other(self):
        self.assertEqual(_normalize_app("Calculator"), "other")

    def test_case_insensitive(self):
        self.assertEqual(_normalize_app("FIREFOX"), "browser")


# ---------------------------------------------------------------------------
# Fallback rule tests
# ---------------------------------------------------------------------------

class TestFallbackRules(unittest.TestCase):

    def test_pinch_any_volume(self):
        self.assertEqual(_fallback_action("PINCH", "other", "auto"), "volume_adjust")

    def test_swipe_right_browser(self):
        self.assertEqual(_fallback_action("SWIPE_RIGHT", "browser", "auto"), "next_tab")

    def test_swipe_right_slides(self):
        self.assertEqual(_fallback_action("SWIPE_RIGHT", "slides", "auto"), "next_slide")

    def test_swipe_right_media(self):
        self.assertEqual(_fallback_action("SWIPE_RIGHT", "media", "auto"), "next_track")

    def test_open_palm_pause(self):
        self.assertEqual(_fallback_action("OPEN_PALM", "code", "auto"), "pause_play")

    def test_fist_click(self):
        self.assertEqual(_fallback_action("FIST", "other", "auto"), "left_click")

    def test_index_up_cursor(self):
        self.assertEqual(_fallback_action("INDEX_UP", "browser", "auto"), "move_cursor")

    def test_mode_volume_overrides(self):
        self.assertEqual(_fallback_action("SWIPE_RIGHT", "other", "volume"), "volume_up")
        self.assertEqual(_fallback_action("SWIPE_LEFT",  "other", "volume"), "volume_down")

    def test_mode_slides_overrides(self):
        self.assertEqual(_fallback_action("SWIPE_RIGHT", "other", "slides"), "next_slide")

    def test_unknown_gesture_returns_none(self):
        self.assertEqual(_fallback_action("WAVE", "other", "auto"), "none")


# ---------------------------------------------------------------------------
# Groq call tests
# ---------------------------------------------------------------------------

class TestGroqCall(unittest.TestCase):

    def test_returns_valid_action(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="next_tab"))]
        )
        result = _call_groq(mock_client, "SWIPE_RIGHT", "browser", "auto")
        self.assertEqual(result, "next_tab")

    def test_returns_none_for_unknown_action(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="open_spotify"))]
        )
        result = _call_groq(mock_client, "SWIPE_RIGHT", "browser", "auto")
        self.assertIsNone(result)

    def test_returns_none_on_exception(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("timeout")
        result = _call_groq(mock_client, "SWIPE_RIGHT", "browser", "auto")
        self.assertIsNone(result)

    def test_returns_none_when_client_is_none(self):
        result = _call_groq(None, "SWIPE_RIGHT", "browser", "auto")
        self.assertIsNone(result)

    def test_strips_extra_words(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="pause_play sure!"))]
        )
        result = _call_groq(mock_client, "OPEN_PALM", "other", "auto")
        self.assertEqual(result, "pause_play")


# ---------------------------------------------------------------------------
# Agent engine integration tests (cache + debounce + fallback path)
# ---------------------------------------------------------------------------

class TestAgentEngine(unittest.TestCase):

    def _make_state(self, gesture="SWIPE_RIGHT", confidence=0.9, mode="auto"):
        state = _AppState()
        state.gesture.name = gesture
        state.gesture.confidence = confidence
        state.current_mode = mode
        return state

    @patch("agent.get_active_app", return_value="browser")
    @patch("agent.execute_action")
    def test_cache_hit_prevents_second_groq_call(self, mock_exec, mock_app):
        state = self._make_state()
        engine = AgentEngine(state)
        engine._groq_client = MagicMock()
        engine._groq_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="next_tab"))]
        )

        # Simulate two identical gesture events
        engine._process_gesture("SWIPE_RIGHT", "browser", "auto", state)
        engine._process_gesture("SWIPE_RIGHT", "browser", "auto", state)

        # Groq called exactly once; second call served from cache
        self.assertEqual(engine._groq_client.chat.completions.create.call_count, 1)

    @patch("agent.get_active_app", return_value="other")
    @patch("agent.execute_action")
    def test_fallback_when_groq_fails(self, mock_exec, mock_app):
        state = self._make_state(gesture="FIST")
        engine = AgentEngine(state)
        engine._groq_client = MagicMock()
        engine._groq_client.chat.completions.create.side_effect = Exception("timeout")

        engine._process_gesture("FIST", "other", "auto", state)

        # Should have called execute_action with the fallback
        mock_exec.assert_called_once_with("left_click", state)

    @patch("agent.get_active_app", return_value="other")
    @patch("agent.execute_action")
    def test_fallback_on_malformed_json(self, mock_exec, mock_app):
        state = self._make_state(gesture="OPEN_PALM")
        engine = AgentEngine(state)
        engine._groq_client = MagicMock()
        engine._groq_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="{invalid_json}"))]
        )

        engine._process_gesture("OPEN_PALM", "other", "auto", state)
        mock_exec.assert_called_once_with("pause_play", state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
