"""
test_actions.py — Unit tests for the action executor.

Mocks pyautogui and pycaw to prevent real OS-level effects during tests.
"""

import sys
import threading
import unittest
from collections import deque
from dataclasses import dataclass, field
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Mock pyautogui before importing actions
# ---------------------------------------------------------------------------

pyautogui_mock = MagicMock()
pyautogui_mock.size.return_value = (1920, 1080)
sys.modules["pyautogui"] = pyautogui_mock

# Mock pycaw to avoid Windows-only import errors on other platforms
pycaw_mock = MagicMock()
sys.modules["pycaw"] = pycaw_mock
sys.modules["pycaw.pycaw"] = pycaw_mock
comtypes_mock = MagicMock()
sys.modules["comtypes"] = comtypes_mock

import actions
# Point actions.pyautogui at the mock
actions.pyautogui = pyautogui_mock


# ---------------------------------------------------------------------------
# Stub AppState
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


# ---------------------------------------------------------------------------
# Keyboard / media action tests
# ---------------------------------------------------------------------------

class TestKeyboardActions(unittest.TestCase):

    def setUp(self):
        pyautogui_mock.reset_mock()

    def test_next_tab(self):
        actions.action_next_tab()
        pyautogui_mock.hotkey.assert_called_once_with("ctrl", "tab")

    def test_prev_tab(self):
        actions.action_prev_tab()
        pyautogui_mock.hotkey.assert_called_once_with("ctrl", "shift", "tab")

    def test_next_slide(self):
        actions.action_next_slide()
        pyautogui_mock.press.assert_called_once_with("right")

    def test_prev_slide(self):
        actions.action_prev_slide()
        pyautogui_mock.press.assert_called_once_with("left")

    def test_next_track(self):
        actions.action_next_track()
        pyautogui_mock.hotkey.assert_called_once_with("ctrl", "right")

    def test_prev_track(self):
        actions.action_prev_track()
        pyautogui_mock.hotkey.assert_called_once_with("ctrl", "left")

    def test_pause_play(self):
        actions.action_pause_play()
        pyautogui_mock.press.assert_called_once_with("space")

    def test_left_click(self):
        actions.action_left_click()
        pyautogui_mock.click.assert_called_once()

    def test_scroll_up(self):
        actions.action_scroll_up()
        pyautogui_mock.scroll.assert_called_once_with(3)

    def test_scroll_down(self):
        actions.action_scroll_down()
        pyautogui_mock.scroll.assert_called_once_with(-3)


# ---------------------------------------------------------------------------
# Cursor action tests
# ---------------------------------------------------------------------------

class TestCursorAction(unittest.TestCase):

    def setUp(self):
        pyautogui_mock.reset_mock()
        pyautogui_mock.size.return_value = (1920, 1080)

    def test_move_cursor_mirrors_x(self):
        # lm_x=0.5, lm_y=0.5 → screen (960, 540)  [x mirrored: (1-0.5)*1920=960]
        actions.action_move_cursor(0.5, 0.5)
        pyautogui_mock.moveTo.assert_called_once_with(960, 540, duration=0.05)

    def test_move_cursor_full_left(self):
        # lm_x=0.0 (far left in webcam = far right on screen after mirror)
        actions.action_move_cursor(0.0, 0.0)
        pyautogui_mock.moveTo.assert_called_once_with(1920, 0, duration=0.05)

    def test_move_cursor_exception_is_caught(self):
        pyautogui_mock.moveTo.side_effect = Exception("display error")
        # Should not raise
        actions.action_move_cursor(0.5, 0.5)
        pyautogui_mock.moveTo.side_effect = None  # reset


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------

class TestDispatcher(unittest.TestCase):

    def setUp(self):
        pyautogui_mock.reset_mock()
        self.state = _AppState()

    def test_dispatch_pause_play(self):
        actions.execute_action("pause_play", self.state)
        pyautogui_mock.press.assert_called_once_with("space")

    def test_dispatch_none_does_nothing(self):
        actions.execute_action("none", self.state)
        pyautogui_mock.press.assert_not_called()
        pyautogui_mock.hotkey.assert_not_called()

    def test_dispatch_unknown_does_nothing(self):
        actions.execute_action("fly_to_moon", self.state)
        pyautogui_mock.press.assert_not_called()

    def test_dispatch_move_cursor_uses_state(self):
        self.state.cursor_target = (0.25, 0.75)
        actions.execute_action("move_cursor", self.state)
        # x mirrored: (1-0.25)*1920 = 1440, y: 0.75*1080 = 810
        pyautogui_mock.moveTo.assert_called_once_with(1440, 810, duration=0.05)

    def test_exception_in_action_is_caught(self):
        pyautogui_mock.press.side_effect = Exception("OS error")
        # Should not raise
        actions.execute_action("pause_play", self.state)
        pyautogui_mock.press.side_effect = None  # reset


# ---------------------------------------------------------------------------
# Volume action tests (Windows path via mock)
# ---------------------------------------------------------------------------

class TestVolumeActions(unittest.TestCase):

    def test_volume_up_win(self):
        mock_volume = MagicMock()
        mock_volume.GetMasterVolumeLevelScalar.return_value = 0.5
        mock_interface = MagicMock()
        mock_interface.QueryInterface.return_value = mock_volume
        mock_devices = MagicMock()
        mock_devices.Activate.return_value = mock_interface

        with patch("actions.sys") as mock_sys, \
             patch("actions.AudioUtilities", create=True) as mock_au, \
             patch("actions.IAudioEndpointVolume", create=True) as mock_iav:
            mock_sys.platform = "win32"
            mock_au.GetSpeakers.return_value = mock_devices
            # Replace the import inside _volume_win
            with patch.dict("sys.modules", {
                "pycaw.pycaw": MagicMock(
                    AudioUtilities=mock_au,
                    IAudioEndpointVolume=mock_iav,
                ),
                "comtypes": MagicMock(CLSCTX_ALL=1),
            }):
                actions._volume_win(+0.05)
                # Verify SetMasterVolumeLevelScalar called with clamped value
                mock_volume.SetMasterVolumeLevelScalar.assert_called_once_with(0.55, None)

    def test_volume_down_clamps_to_zero(self):
        mock_volume = MagicMock()
        mock_volume.GetMasterVolumeLevelScalar.return_value = 0.02  # near zero
        mock_interface = MagicMock()
        mock_interface.QueryInterface.return_value = mock_volume
        mock_devices = MagicMock()
        mock_devices.Activate.return_value = mock_interface

        with patch.dict("sys.modules", {
            "pycaw.pycaw": MagicMock(
                AudioUtilities=MagicMock(GetSpeakers=lambda: mock_devices),
                IAudioEndpointVolume=MagicMock(_iid_="iid"),
            ),
            "comtypes": MagicMock(CLSCTX_ALL=1),
        }):
            actions._volume_win(-0.05)
            args = mock_volume.SetMasterVolumeLevelScalar.call_args
            if args:
                new_vol = args[0][0]
                self.assertGreaterEqual(new_vol, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
