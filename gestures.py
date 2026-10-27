"""Gesture vocabulary and an exact consecutive-frame command gate."""
from dataclasses import dataclass
import math


# gesture -> (Vietnamese gesture name, command, Vietnamese command label)
GESTURES = {
    "Open_Palm": ("Xòe bàn tay", "START", "BẮT ĐẦU"),
    "Closed_Fist": ("Nắm tay", "STOP", "DỪNG"),
    "Pointing_Up": ("Trỏ ngón tay lên", "MOVE_UP", "ĐI LÊN"),
    "Thumb_Up": ("Giơ ngón cái lên", "ZOOM_IN", "PHÓNG TO"),
    "Thumb_Down": ("Úp ngón cái xuống", "ZOOM_OUT", "THU NHỎ"),
    "Victory": ("Hai ngón chữ V", "CHANGE_COLOR", "ĐỔI MÀU"),
    "ILoveYou": ("Ngón cái + trỏ + út", "PARTICLES", "BUNG LẤP LÁNH"),
}


@dataclass(frozen=True)
class GateState:
    gesture: str | None
    frames: int
    required: int
    stable: bool
    event: str | None = None


class StableGestureGate:
    """Emit once on >=N equal valid frames, then wait for a new stable gesture.

    Five consecutive neutral/unknown/missing frames rearm the same gesture.
    A single unknown or low-confidence frame resets the current streak, but
    does not rearm a held command. Long capture gaps also reset the streak.
    """

    def __init__(self, frames=5, confidence=0.65, max_gap=0.5):
        if not isinstance(frames, int) or frames < 5:
            raise ValueError("Phai on dinh it nhat 5 frame.")
        if not 0 < confidence <= 1:
            raise ValueError("Confidence phai trong (0, 1].")
        self.required = frames
        self.confidence = confidence
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.candidate = None
        self.count = 0
        self.last_emitted = None
        self.neutral_count = 0
        self.previous_time = None

    def update(self, gesture, confidence, now=None, identity=None):
        if now is not None and self.previous_time is not None:
            if now <= self.previous_time or now - self.previous_time > self.max_gap:
                self.candidate, self.count = None, 0
        if now is not None:
            self.previous_time = now
        valid = gesture in GESTURES and math.isfinite(confidence) and confidence >= self.confidence
        if not valid:
            self.candidate, self.count = None, 0
            self.neutral_count += 1
            if self.neutral_count >= self.required:
                self.last_emitted = None
            return GateState(None, 0, self.required, False)
        self.neutral_count = 0
        # Handedness prevents left/right hands contributing to the same streak.
        candidate = (gesture, identity)
        if self.candidate == candidate:
            self.count = min(self.count + 1, self.required)
        else:
            self.candidate, self.count = candidate, 1
        stable = self.count >= self.required
        event = None
        if stable and gesture != self.last_emitted:
            self.last_emitted = gesture
            event = GESTURES[gesture][1]
        return GateState(gesture, self.count, self.required, stable, event)
