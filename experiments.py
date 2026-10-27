"""Labeled real-webcam trials and comparable, persistent CSV metrics."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import csv
import math
import statistics

from gestures import GESTURES

CONDITIONS = {
    "normal": "Ánh sáng bình thường",
    "low_light": "Ánh sáng yếu",
    "occluded": "Tay bị che khuất",
    "near": "Gần · khoảng 30 cm",
    "medium": "Vừa · khoảng 60 cm",
    "far": "Xa · khoảng 100 cm",
}


@dataclass
class TrialMetrics:
    expected: str
    frames: int = 0
    detected: int = 0
    correct_frames: int = 0
    correct_events: int = 0
    wrong_events: int = 0
    confidence_sum: float = 0.0
    brightness_sum: float = 0.0
    area_sum: float = 0.0
    first_correct_seconds: float | None = None
    confusion: dict = field(default_factory=dict)

    def observe(self, observation, gate, elapsed):
        self.frames += 1
        self.detected += int(bool(observation.landmarks))
        correct = gate.gesture == self.expected
        self.correct_frames += int(correct)
        self.confidence_sum += observation.confidence
        self.brightness_sum += observation.brightness
        self.area_sum += observation.area
        predicted = gate.gesture or "Unknown"
        self.confusion[predicted] = self.confusion.get(predicted, 0) + 1
        if gate.event:
            if gate.event == GESTURES[self.expected][1]:
                self.correct_events += 1
                if self.first_correct_seconds is None:
                    self.first_correct_seconds = elapsed
            else:
                self.wrong_events += 1

    def row(self, condition, duration, source="webcam_real"):
        n = max(self.frames, 1)
        return {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": source,
            "condition": condition,
            "expected_gesture": self.expected,
            "duration_seconds": round(duration, 4),
            "frames": self.frames,
            "detected_frames": self.detected,
            "correct_frames": self.correct_frames,
            "correct_events": self.correct_events,
            "wrong_events": self.wrong_events,
            "success": int(self.correct_events > 0 and self.wrong_events == 0),
            "frame_accuracy_percent": round(100 * self.correct_frames / n, 2),
            "detection_percent": round(100 * self.detected / n, 2),
            "first_correct_seconds": "" if self.first_correct_seconds is None else round(self.first_correct_seconds, 4),
            "mean_confidence": round(self.confidence_sum / n, 4),
            "mean_brightness_0_255": round(self.brightness_sum / n, 2),
            "mean_hand_area_percent": round(100 * self.area_sum / n, 2),
            "fps": round(self.frames / duration, 2) if duration > 0 else 0,
        }


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def summarize(rows, expected=None):
    result = {}
    for condition in CONDITIONS:
        matching = [r for r in rows if r["condition"] == condition and (expected is None or r["expected_gesture"] == expected)]
        if not matching:
            continue
        frames = sum(int(r["frames"]) for r in matching)
        latencies = [float(r["first_correct_seconds"]) for r in matching if r["first_correct_seconds"] != ""]
        result[condition] = {
            "trials": len(matching),
            "frame_accuracy": 100 * sum(int(r["correct_frames"]) for r in matching) / max(1, frames),
            "detection": 100 * sum(int(r["detected_frames"]) for r in matching) / max(1, frames),
            "success": 100 * sum(int(r["success"]) for r in matching) / len(matching),
            "wrong_events": sum(int(r["wrong_events"]) for r in matching),
            "latency": statistics.mean(latencies) if latencies else None,
        }
    return result


class ExperimentLab:
    def __init__(self, path, seconds=4.0):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Thoi luong thu nghiem phai duong.")
        self.path = Path(path)
        self.seconds = seconds
        self.rows = read_rows(self.path)
        self.condition = "normal"
        self.expected = "Open_Palm"
        self.active = None
        self.message = "Chọn điều kiện, chọn cử chỉ rồi bấm T để đo"

    def start(self, now):
        if self.active is not None:
            return
        self.active = {"ready": now + 2, "end": now + 2 + self.seconds,
                       "condition": self.condition, "expected": self.expected,
                       "started": False, "metrics": TrialMetrics(self.expected)}

    def cancel(self):
        self.active = None
        self.message = "Đã hủy lượt đo; không ghi kết quả dở"

    def tick(self, now):
        """Returns True exactly when recording begins; reset the gate then."""
        if self.active is None:
            return False
        if now >= self.active["end"]:
            active = self.active
            if active["metrics"].frames:
                row = active["metrics"].row(active["condition"], self.seconds)
                append_csv(self.path, row)
                self.rows.append(row)
                self.message = "Đã lưu lượt đo vào results/trials.csv"
            else:
                self.message = "Không có khung hình trong lượt đo; không lưu"
            self.active = None
            return False
        if now >= self.active["ready"] and not self.active["started"]:
            self.active["started"] = True
            return True
        return False

    def observe(self, observation, gate, now):
        if self.active and self.active["started"]:
            self.active["metrics"].observe(observation, gate, now - self.active["ready"])

    def status(self, now):
        if not self.active:
            return self.message
        expected = GESTURES[self.active["expected"]][0]
        if not self.active["started"]:
            return f"Chuẩn bị {max(1, math.ceil(self.active['ready'] - now))}s · {expected}"
        return f"ĐANG ĐO {max(0, self.active['end'] - now):.1f}s · giữ {expected}"
