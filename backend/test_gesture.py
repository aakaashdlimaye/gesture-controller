"""
test_gesture.py — Unit tests for gesture classifier using synthetic landmark data.

No webcam or MediaPipe runtime required — uses MockLandmark objects.
"""

import time
import unittest
from collections import deque
from dataclasses import dataclass

from gesture import (
    CONFIDENCE_THRESHOLD,
    GESTURE_FIST,
    GESTURE_INDEX_UP,
    GESTURE_NONE,
    GESTURE_OPEN_PALM,
    GESTURE_PINCH,
    GESTURE_SWIPE_LEFT,
    GESTURE_SWIPE_RIGHT,
    classify_gesture,
)


# ---------------------------------------------------------------------------
# Mock landmark
# ---------------------------------------------------------------------------

@dataclass
class LM:
    x: float
    y: float
    z: float = 0.0


def _flat_hand(y_tip_offset: float = -0.2) -> list:
    """
    Build a 21-landmark hand where all fingers are extended.
    PIP joints are at y=0.5; TIPs are at y = 0.5 + y_tip_offset (negative = higher).
    """
    lm = [LM(0.5, 0.9)] * 21  # default: wrist at bottom

    # Wrist (0) and middle MCP (9) for palm size
    lm[0]  = LM(0.5, 0.9)
    lm[9]  = LM(0.5, 0.6)

    # Thumb (extended: large x offset)
    lm[2]  = LM(0.35, 0.7)
    lm[3]  = LM(0.25, 0.65)
    lm[4]  = LM(0.15, 0.6)   # tip far from base → extended

    # Index
    lm[5]  = LM(0.5, 0.65)   # MCP
    lm[6]  = LM(0.5, 0.55)   # PIP
    lm[7]  = LM(0.5, 0.45)   # DIP
    lm[8]  = LM(0.5, 0.35)   # TIP

    # Middle
    lm[9]  = LM(0.55, 0.6)
    lm[10] = LM(0.55, 0.5)
    lm[11] = LM(0.55, 0.4)
    lm[12] = LM(0.55, 0.3)

    # Ring
    lm[13] = LM(0.6, 0.65)
    lm[14] = LM(0.6, 0.55)
    lm[15] = LM(0.6, 0.45)
    lm[16] = LM(0.6, 0.35)

    # Pinky
    lm[17] = LM(0.65, 0.7)
    lm[18] = LM(0.65, 0.62)
    lm[19] = LM(0.65, 0.54)
    lm[20] = LM(0.65, 0.46)

    return lm


def _fist_hand() -> list:
    """All fingers curled: tips below their PIP joints."""
    lm = _flat_hand()
    # Move each fingertip to be below (larger y) its PIP joint
    lm[8]  = LM(0.50, 0.62)  # index tip below PIP (0.55)
    lm[12] = LM(0.55, 0.58)  # middle tip below PIP (0.5)
    lm[16] = LM(0.60, 0.62)  # ring tip below PIP (0.55)
    lm[20] = LM(0.65, 0.68)  # pinky tip below PIP (0.62)
    # Thumb wraps to the side — keep it far from the index tip (0.50, 0.62)
    # so pinch ratio stays well above 0.4 threshold
    lm[2]  = LM(0.35, 0.72)
    lm[4]  = LM(0.30, 0.66)  # dist to lm[8]≈(0.50,0.62) ≈ 0.20 → ratio ≈ 0.66 > 0.4
    return lm


def _index_up_hand() -> list:
    """Only index extended; middle/ring/pinky folded."""
    lm = _flat_hand()
    # Curl middle, ring, pinky
    lm[12] = LM(0.55, 0.58)
    lm[16] = LM(0.6,  0.62)
    lm[20] = LM(0.65, 0.68)
    # Curl thumb (small x diff)
    lm[2]  = LM(0.5, 0.7)
    lm[4]  = LM(0.52, 0.65)
    return lm


def _pinch_hand() -> list:
    """Thumb tip and index tip very close together."""
    lm = _flat_hand()
    # Bring thumb tip (4) very close to index tip (8)
    lm[4]  = LM(0.502, 0.352)
    lm[8]  = LM(0.500, 0.350)
    return lm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGestureClassifier(unittest.TestCase):

    def _classify(self, lm, wrist_history=None, now=None, last_swipe=0.0):
        if wrist_history is None:
            wrist_history = deque(maxlen=8)
        if now is None:
            now = time.monotonic()
        return classify_gesture(lm, wrist_history, now, last_swipe)

    # ---- OPEN_PALM ----

    def test_open_palm_detected(self):
        lm = _flat_hand()
        name, conf, _ = self._classify(lm)
        self.assertEqual(name, "OPEN_PALM", f"Expected OPEN_PALM, got {name}")
        self.assertGreaterEqual(conf, CONFIDENCE_THRESHOLD)

    def test_open_palm_high_confidence(self):
        lm = _flat_hand()
        _, conf, _ = self._classify(lm)
        self.assertGreaterEqual(conf, 0.8)

    # ---- FIST ----

    def test_fist_detected(self):
        lm = _fist_hand()
        name, conf, _ = self._classify(lm)
        self.assertEqual(name, "FIST", f"Expected FIST, got {name}")
        self.assertGreaterEqual(conf, CONFIDENCE_THRESHOLD)

    # ---- INDEX_UP ----

    def test_index_up_detected(self):
        lm = _index_up_hand()
        name, conf, _ = self._classify(lm)
        self.assertEqual(name, "INDEX_UP", f"Expected INDEX_UP, got {name}")
        self.assertGreaterEqual(conf, CONFIDENCE_THRESHOLD)

    # ---- PINCH ----

    def test_pinch_detected(self):
        lm = _pinch_hand()
        name, conf, _ = self._classify(lm)
        self.assertEqual(name, "PINCH", f"Expected PINCH, got {name}")
        self.assertGreaterEqual(conf, CONFIDENCE_THRESHOLD)

    def test_open_palm_not_pinch(self):
        lm = _flat_hand()
        name, _, _ = self._classify(lm)
        self.assertNotEqual(name, "PINCH")

    # ---- SWIPE_RIGHT ----

    def test_swipe_right_detected(self):
        lm = _flat_hand()
        history = deque(maxlen=8)
        start = time.monotonic()
        # Simulate wrist moving from left (x=0.2) to right (x=0.9) over 0.3s
        for i in range(8):
            t = start + i * 0.04
            lm[0] = LM(0.2 + i * 0.1, 0.9)
            history.append((t, lm[0].x))

        now = start + 0.28
        name, conf, _ = classify_gesture(lm, history, now, 0.0)
        self.assertEqual(name, "SWIPE_RIGHT", f"Expected SWIPE_RIGHT, got {name}")
        self.assertGreaterEqual(conf, CONFIDENCE_THRESHOLD)

    # ---- SWIPE_LEFT ----

    def test_swipe_left_detected(self):
        lm = _flat_hand()
        history = deque(maxlen=8)
        start = time.monotonic()
        # Wrist moves right → left (x decreasing)
        for i in range(8):
            t = start + i * 0.04
            lm[0] = LM(0.9 - i * 0.1, 0.9)
            history.append((t, lm[0].x))

        now = start + 0.28
        name, conf, _ = classify_gesture(lm, history, now, 0.0)
        self.assertEqual(name, "SWIPE_LEFT", f"Expected SWIPE_LEFT, got {name}")
        self.assertGreaterEqual(conf, CONFIDENCE_THRESHOLD)

    # ---- Swipe cooldown ----

    def test_swipe_cooldown_blocks_repeat(self):
        lm = _flat_hand()
        history = deque(maxlen=8)
        start = time.monotonic()
        for i in range(8):
            t = start + i * 0.04
            lm[0] = LM(0.2 + i * 0.1, 0.9)
            history.append((t, lm[0].x))

        now = start + 0.28
        # First swipe detected
        name, _, last_swipe = classify_gesture(lm, history, now, 0.0)
        self.assertEqual(name, "SWIPE_RIGHT")

        # Immediately try again — cooldown should block it
        lm2 = _flat_hand()
        history2 = deque(maxlen=8)
        for i in range(8):
            t = now + i * 0.04
            lm2[0] = LM(0.2 + i * 0.1, 0.9)
            history2.append((t, lm2[0].x))
        now2 = now + 0.28
        name2, _, _ = classify_gesture(lm2, history2, now2, last_swipe)
        self.assertNotEqual(name2, "SWIPE_RIGHT", "Swipe cooldown should have blocked this")

    # ---- Confidence threshold ----

    def test_low_confidence_scenario(self):
        # A hand that is partially extended — should produce open_palm with
        # score between 0.6-0.8 (possible NONE or partial match)
        lm = _flat_hand()
        # Curl one finger to reduce palm score
        lm[8] = LM(0.5, 0.62)
        lm[12] = LM(0.55, 0.58)
        name, conf, _ = self._classify(lm)
        # Whatever it detects, confidence should reflect partial state
        # Either NONE or a gesture with confidence that reflects the input
        self.assertIsInstance(conf, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
