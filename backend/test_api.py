"""
test_api.py — Integration tests for the FastAPI server endpoints.

Uses httpx TestClient; mocks GestureDetector and AgentEngine to avoid
requiring a real webcam or Groq API key.
"""

import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

# Mock pyautogui before any import chain reaches actions.py
pyautogui_mock = MagicMock()
pyautogui_mock.size.return_value = (1920, 1080)
sys.modules.setdefault("pyautogui", pyautogui_mock)
sys.modules.setdefault("pycaw", MagicMock())
sys.modules.setdefault("pycaw.pycaw", MagicMock())
sys.modules.setdefault("comtypes", MagicMock())
sys.modules.setdefault("mediapipe", MagicMock())
sys.modules.setdefault("cv2", MagicMock())
sys.modules.setdefault("groq", MagicMock())

from fastapi.testclient import TestClient

import main as app_module
from main import app, state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset_state():
    """Reset shared state before each test."""
    with state.lock:
        state.gesture.name = "NONE"
        state.gesture.confidence = 0.0
        state.fps = 0.0
        state.webcam_active = False
        state.active_app = "unknown"
        state.last_action = "none"
        state.current_mode = "auto"
        state.action_log.clear()
    state.stop_event.clear()


client = TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

class TestStatusEndpoint:

    def setup_method(self):
        reset_state()

    def test_status_returns_200(self):
        resp = client.get("/status")
        assert resp.status_code == 200

    def test_status_shape(self):
        resp = client.get("/status")
        data = resp.json()
        assert "gesture" in data
        assert "confidence" in data
        assert "active_app" in data
        assert "last_action" in data
        assert "fps" in data
        assert "mode" in data
        assert "webcam_active" in data

    def test_status_defaults(self):
        resp = client.get("/status")
        data = resp.json()
        assert data["gesture"] == "NONE"
        assert data["confidence"] == 0.0
        assert data["mode"] == "auto"
        assert data["webcam_active"] is False


# ---------------------------------------------------------------------------
# /mode/{mode}
# ---------------------------------------------------------------------------

class TestModeEndpoint:

    def setup_method(self):
        reset_state()

    def test_valid_mode_auto(self):
        resp = client.post("/mode/auto")
        assert resp.status_code == 200
        assert resp.json()["mode"] == "auto"

    def test_valid_mode_volume(self):
        resp = client.post("/mode/volume")
        assert resp.status_code == 200

    def test_valid_mode_slides(self):
        resp = client.post("/mode/slides")
        assert resp.status_code == 200

    def test_valid_mode_cursor(self):
        resp = client.post("/mode/cursor")
        assert resp.status_code == 200

    def test_invalid_mode_returns_400(self):
        resp = client.post("/mode/fly")
        assert resp.status_code == 400

    def test_mode_persists_in_status(self):
        client.post("/mode/slides")
        resp = client.get("/status")
        assert resp.json()["mode"] == "slides"


# ---------------------------------------------------------------------------
# /start and /stop
# ---------------------------------------------------------------------------

class TestStartStop:

    def setup_method(self):
        reset_state()

    @patch("gesture.GestureDetector")
    @patch("agent.AgentEngine")
    def test_start_returns_started(self, mock_agent, mock_gesture):
        mock_gesture.return_value.start = MagicMock()
        mock_gesture.return_value._thread = MagicMock()
        mock_agent.return_value.start = MagicMock()
        mock_agent.return_value._thread = MagicMock()

        resp = client.post("/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

    @patch("gesture.GestureDetector")
    @patch("agent.AgentEngine")
    def test_double_start_returns_already_running(self, mock_agent, mock_gesture):
        mock_gesture.return_value.start = MagicMock()
        mock_gesture.return_value._thread = MagicMock()
        mock_agent.return_value.start = MagicMock()
        mock_agent.return_value._thread = MagicMock()

        with state.lock:
            state.webcam_active = True  # Simulate already running

        resp = client.post("/start")
        assert resp.json()["status"] == "already_running"

    def test_stop_returns_stopped(self):
        resp = client.post("/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_stop_sets_stop_event(self):
        client.post("/stop")
        assert state.stop_event.is_set()


# ---------------------------------------------------------------------------
# /logs
# ---------------------------------------------------------------------------

class TestLogsEndpoint:

    def setup_method(self):
        reset_state()

    def test_logs_empty_initially(self):
        resp = client.get("/logs")
        assert resp.status_code == 200
        assert resp.json()["logs"] == []

    def test_logs_returns_entries(self):
        with state.lock:
            for i in range(5):
                state.action_log.appendleft(f"entry {i}")
        resp = client.get("/logs")
        logs = resp.json()["logs"]
        assert len(logs) == 5

    def test_logs_capped_at_50(self):
        with state.lock:
            for i in range(60):
                state.action_log.appendleft(f"entry {i}")
        resp = client.get("/logs")
        logs = resp.json()["logs"]
        assert len(logs) <= 50
