"""Confidence-gated arm geometry, stable actions and planar robot kinematics."""
from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

ARM_INDICES = {"left": (11, 13, 15), "right": (12, 14, 16)}
ACTION_NAMES = {"RAISE": "Giơ tay", "LOWER": "Hạ tay", "WAVE": "Vẫy tay",
                "HOLD": "Giữ tư thế", "WAIT": "Đang xác nhận", "NO_POSE": "Chưa đủ điểm mốc"}


def joint_angle(a, b, c):
    """Angle ABC in degrees, with invalid/degenerate geometry rejected."""
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    if a.shape != b.shape or b.shape != c.shape or a.ndim != 1 or a.size not in (2, 3):
        raise ValueError("Three equally sized 2D/3D points are required")
    u, v = a-b, c-b
    norms = np.linalg.norm(u)*np.linalg.norm(v)
    if not np.isfinite([*u, *v]).all() or not math.isfinite(norms) or norms < 1e-9:
        raise ValueError("Invalid or coincident arm points")
    return math.degrees(math.acos(float(np.clip(np.dot(u, v)/norms, -1, 1))))


def robot_points(shoulder_heading, elbow_angle, bend_sign, origin=(0, 0), lengths=(1, 1)):
    """Forward kinematics; elbow angle uses the same interior-angle convention."""
    shoulder = np.asarray(origin, dtype=float)
    heading = math.radians(shoulder_heading)
    forearm = heading + bend_sign*math.radians(180-elbow_angle)
    elbow = shoulder + lengths[0]*np.array((math.cos(heading), math.sin(heading)))
    wrist = elbow + lengths[1]*np.array((math.cos(forearm), math.sin(forearm)))
    return np.array((shoulder, elbow, wrist))


@dataclass
class Observation:
    landmarks: list = field(default_factory=list)
    world_landmarks: list = field(default_factory=list)
    width: int = 640
    height: int = 480


@dataclass(frozen=True)
class ArmSample:
    side: str
    points: np.ndarray
    angle: float
    heading: float
    bend: int
    confidence: float
    elevation: float
    lateral: float


def arm_sample(observation, side, mirror=True, threshold=.65):
    if len(observation.landmarks) < 33:
        return None
    selected = [observation.landmarks[i] for i in ARM_INDICES[side]]
    scores=np.array([(getattr(p,"visibility",0) or 0,getattr(p,"presence",0) or 0) for p in selected],dtype=float)
    if not np.isfinite(scores).all():
        return None
    confidence=float(scores.min())
    if confidence < threshold:
        return None
    normalized = np.array([(p.x, p.y) for p in selected], dtype=float)
    if not np.isfinite(normalized).all() or (normalized < 0).any() or (normalized > 1).any():
        return None
    if mirror:
        normalized[:, 0] = 1-normalized[:, 0]
    # Aspect correction is essential: normalized X/Y have different units.
    points = normalized*np.array((observation.width, observation.height))
    upper, forearm = points[1]-points[0], points[2]-points[1]
    lengths = (np.linalg.norm(upper), np.linalg.norm(forearm))
    if min(lengths) < max(4, min(observation.width, observation.height)*.012):
        return None
    try:
        angle = joint_angle(*points)
    except ValueError:
        return None
    cross = upper[0]*forearm[1]-upper[1]*forearm[0]
    heading = math.degrees(math.atan2(upper[1], upper[0]))
    return ArmSample(side, points, angle, heading, 1 if cross >= 0 else -1, confidence,
                     float((points[0, 1]-points[2, 1])/sum(lengths)),
                     float((points[2, 0]-points[1, 0])/lengths[1]))


class WaveDetector:
    """Require three lateral reversals, sufficient excursion and a raised wrist."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.turns = deque()
        self.direction = 0
        self.extreme = None
        self.last_time = None
        self.filtered = None
        self.active_until = -math.inf

    def update(self, lateral, eligible, now):
        if not eligible or (self.last_time is not None and (now <= self.last_time or now-self.last_time > .4)):
            self.reset()
        if not eligible:
            return False
        dt = 1/30 if self.last_time is None else now-self.last_time
        self.last_time = now
        self.filtered = lateral if self.filtered is None else self.filtered+(1-math.exp(-dt/.045))*(lateral-self.filtered)
        value = self.filtered
        while self.turns and now-self.turns[0][0] > 1.6:
            self.turns.popleft()
        if self.extreme is None:
            self.extreme = (now, value)
        delta = value-self.extreme[1]
        fresh_turn=False
        if self.direction == 0:
            if abs(delta) >= .22:
                self.direction = 1 if delta > 0 else -1
                self.extreme = (now, value)
        elif delta*self.direction > 0:
            self.extreme = (now, value)
        elif delta*self.direction < -.22:
            if not self.turns or self.extreme[0]-self.turns[-1][0] >= .12:
                self.turns.append(self.extreme)
                fresh_turn=True
            self.direction *= -1
            self.extreme = (now, value)
        if fresh_turn and len(self.turns) >= 3:
            amplitudes = [v for _, v in self.turns]
            if max(amplitudes)-min(amplitudes) >= .35:
                self.active_until = now+.7
        return now < self.active_until


@dataclass(frozen=True)
class MotionState:
    valid: bool = False
    side: str | None = None
    angle: float | None = None
    raw_angle: float | None = None
    shoulder_heading: float = 45.0
    robot_elbow: float = 100.0
    bend: int = -1
    confidence: float = 0.0
    action: str = "NO_POSE"
    candidate: str = "NO_POSE"
    frames: int = 0
    required: int = 5
    event: str | None = None
    points: np.ndarray | None = None


class PoseController:
    def __init__(self, required=5, threshold=.65):
        if required < 5:
            raise ValueError("At least five stable frames are required")
        self.required, self.threshold = required, threshold
        self.reset()

    def reset(self):
        self.side = None
        self.preference = None
        self.mirror = None
        self.last_time = self.last_valid = None
        self.heading = None
        self.angle = None
        self.bend = -1
        self.candidate = None
        self.frames = 0
        self.action = "WAIT"
        self.wave = WaveDetector()

    def update(self, observation, now, preference="auto", mirror=True):
        if preference not in ("auto", "left", "right"):
            raise ValueError("Unknown arm selection")
        if (self.preference, self.mirror) != (preference, mirror):
            self.reset()
        self.preference, self.mirror = preference, mirror
        dt = 1/30 if self.last_time is None else max(0, now-self.last_time)
        if dt > .4 or (self.last_time is not None and now <= self.last_time):
            self.wave.reset()
            self.frames = 0
            self.candidate = None
            self.action = "WAIT"
        self.last_time = now
        samples = {side: arm_sample(observation, side, mirror, self.threshold) for side in ARM_INDICES}
        if preference != "auto":
            chosen = preference
        elif self.side is not None and (samples[self.side] is not None or (self.last_valid is not None and now-self.last_valid < .65)):
            chosen = self.side
        else:
            available = [s for s in samples.values() if s is not None]
            chosen = max(available, key=lambda s: (s.confidence, s.side == "right")).side if available else self.side
        sample = samples.get(chosen)
        if sample is None:
            self.candidate, self.frames, self.action = None, 0, "WAIT"
            self.wave.reset()
            return MotionState(side=self.side, shoulder_heading=self.heading if self.heading is not None else 45,
                               robot_elbow=self.angle if self.angle is not None else 100, bend=self.bend,
                               required=self.required)
        if chosen != self.side:
            self.wave.reset()
            self.frames, self.candidate, self.action = 0, None, "WAIT"
            self.heading = self.angle = None
        self.side, self.last_valid = chosen, now
        if self.heading is None:
            self.heading, self.angle = sample.heading, sample.angle
        else:
            alpha = 1-math.exp(-min(dt, .2)/.075)
            delta = (sample.heading-self.heading+180)%360-180
            self.heading += float(np.clip(alpha*delta, -540*dt, 540*dt))
            self.angle += float(np.clip(alpha*(sample.angle-self.angle), -540*dt, 540*dt))
        if 5 < sample.angle < 175:
            self.bend = sample.bend
        raised = sample.elevation > (.04 if self.action in ("RAISE", "WAVE") else .12)
        lowered = sample.elevation < (-.22 if self.action == "LOWER" else -.35)
        waving = self.wave.update(sample.lateral, raised and 15 < sample.angle < 175, now)
        candidate = "WAVE" if waving else "RAISE" if raised else "LOWER" if lowered else "HOLD"
        if candidate == self.candidate:
            self.frames = min(self.required, self.frames+1)
        else:
            self.candidate, self.frames = candidate, 1
        event = None
        if self.frames >= self.required and self.action != candidate:
            self.action = candidate
            if candidate != "HOLD":
                event = candidate
        return MotionState(True, chosen, self.angle, sample.angle, self.heading, self.angle, self.bend,
                           sample.confidence, self.action, candidate, self.frames, self.required, event, sample.points)
