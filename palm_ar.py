"""Palm-anchored AR cube and a gesture-gated, sparkling 3D explosion."""
import math

import cv2
import numpy as np


class PalmAR:
    BURST_SECONDS = 2.4

    def __init__(self):
        self.anchor = None
        self.span = None
        self.length = None
        self.identity = None
        self.last_wrist = None
        self.visible = False
        self.spawned = False
        self.armed = False
        self.phase = "waiting"
        self.burst_age = 0.0
        self.positions = np.empty((0, 3))
        self.velocities = np.empty((0, 3))
        self.twinkle = np.empty(0)
        self.burst_anchor = None
        self.burst_span = None
        self.burst_length = None
        self.time = 0.0
        self.appearance = 0.0
        self.burst_count = 0
        self.control_scale = self.burst_scale = 1.0
        self.control_height = self.burst_height = 0.0
        self.random = np.random.default_rng(42)

    def lose_hand(self):
        self.anchor = self.span = self.length = None
        self.burst_anchor = self.burst_span = self.burst_length = None
        self.last_wrist = self.identity = None
        self.visible = self.spawned = self.armed = False
        self.phase = "waiting"
        self.positions = np.empty((0, 3))
        self.appearance = 0.0

    def explode(self):
        if not self.visible or not self.spawned or self.phase == "burst":
            return False
        count = 140
        self.positions = self.random.uniform(-0.52, 0.52, (count, 3))
        self.positions[:, 1] += 0.65
        direction = self.random.normal(size=(count, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        self.velocities = direction * self.random.uniform(1.2, 4.0, (count, 1))
        self.velocities[:, 1] += 1.5
        self.twinkle = self.random.uniform(0, 2 * math.pi, count)
        # Released particles stay at the release position while the hand moves.
        self.burst_anchor = self.anchor.copy()
        self.burst_span = self.span.copy()
        self.burst_length = self.length.copy()
        self.burst_scale = self.control_scale
        self.burst_height = self.control_height
        self.burst_age = 0.0
        self.armed = False
        self.phase = "burst"
        self.burst_count += 1
        return True

    def update(self, landmarks, gate, dt, identity=None, scene=None):
        dt = min(max(float(dt), 0.0), 0.25)
        self.time += dt
        if scene is not None:
            self.control_scale, self.control_height = scene.scale, scene.height
        if len(landmarks) < 21:
            self.lose_hand()
            return
        points = np.array([(p.x, p.y) for p in landmarks], dtype=float)
        if not np.isfinite(points).all():
            self.lose_hand()
            return
        wrist = points[0]
        switched = self.identity is not None and identity is not None and identity != self.identity
        jumped = self.last_wrist is not None and np.linalg.norm(wrist - self.last_wrist) > 0.3
        if switched or jumped:
            self.lose_hand()
        self.identity, self.last_wrist = identity, wrist.copy()
        center = points[[0, 5, 9, 13, 17]].mean(axis=0)
        span, length = points[5] - points[17], points[9] - points[0]
        if max(np.linalg.norm(span), np.linalg.norm(length)) < 0.015:
            self.lose_hand()
            return
        if self.anchor is None:
            self.anchor, self.span, self.length = center, span, length
        else:
            alpha = 1 - math.exp(-dt / 0.06)
            self.anchor += alpha * (center - self.anchor)
            self.span += alpha * (span - self.span)
            self.length += alpha * (length - self.length)
        self.visible = True
        self.appearance = min(1.0, self.appearance + dt * 3)
        if self.phase == "burst":
            self.burst_age += dt
            self.positions += self.velocities * dt
            self.velocities[:, 1] -= 1.8 * dt
            self.velocities *= math.exp(-0.18 * dt)
            if self.burst_age >= self.BURST_SECONDS:
                self.positions = np.empty((0, 3))
                self.phase = "holding"
                self.appearance = 0.0
        # No action from raw labels: all interactions require the stable gate.
        if not gate.stable:
            return
        if gate.gesture == "Open_Palm":
            if not self.spawned:
                self.spawned = True
                self.phase = "holding"
                self.appearance = 0.0
            elif self.armed:
                self.explode()
        elif gate.gesture == "Closed_Fist" and self.spawned and self.phase != "burst":
            self.armed = True
            self.phase = "charged"
        elif gate.event == "PARTICLES":
            self.explode()

    @property
    def hint(self):
        if not self.visible or not self.spawned:
            return "Xòe tay ổn định để hứng khối lập phương"
        if self.phase == "burst":
            return "Đang bung tỏa · các hạt lấp lánh tan dần"
        if self.armed:
            return "Đã tích năng lượng · xòe tay để bung tỏa!"
        return "Nắm tay rồi xòe ra để khối bung lấp lánh"

    @staticmethod
    def _placement(anchor, span, length, transform):
        ox, oy, width, height = transform
        dimensions = np.array((width, height))
        center = np.array((ox, oy)) + anchor * dimensions
        size = max(np.linalg.norm(span * dimensions), 0.65 * np.linalg.norm(length * dimensions))
        return center, float(np.clip(size, 10, min(width, height) * 0.38))

    @staticmethod
    def _project(points, center, size):
        points = np.asarray(points, dtype=float)
        depth = np.maximum(2.0, 6.5 + points[:, 2])
        factor = size * 3.3 / depth
        return np.column_stack((center[0] + points[:, 0] * factor,
                                center[1] - points[:, 1] * factor))

    def overlay(self, frame, transform, scene):
        """Draw only inside the camera's active letterboxed image region."""
        if not self.visible or not self.spawned or self.anchor is None:
            return frame
        center, size = self._placement(self.anchor, self.span, self.length, transform)
        base = frame.copy()
        ink = np.zeros_like(base)
        solid_mask = np.zeros(base.shape[:2], dtype=np.uint8)
        color = np.array(scene.colors[scene.color_index][::-1], dtype=float)
        size *= scene.scale
        center[1] -= scene.height * size * 0.5

        if self.phase != "burst":
            # A contact halo and soft shadow place the cube on the palm.
            cv2.ellipse(ink, tuple(center.astype(int)), (max(3, int(size * 0.5)), max(2, int(size * 0.13))),
                        0, 0, 360, tuple(int(c * 0.5) for c in color), 2, cv2.LINE_AA)
            charge = 0.78 + 0.05 * math.sin(self.time * 16) if self.armed else 1.0
            growth = 0.25 + 0.75 * min(self.appearance * 2, 1)
            vertices = scene.vertices.copy() * (0.65 * charge * growth)
            vertices[:, 1] += 0.65 * charge * growth
            angle = scene.angle
            ry = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                           [-math.sin(angle), 0, math.cos(angle)]])
            # Rotate around the cube's center; its bottom remains on the hand.
            vertices[:, 1] -= 0.65 * charge * growth
            vertices = vertices @ ry.T
            tilt = -0.25
            rx = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)],
                           [0, math.sin(tilt), math.cos(tilt)]])
            vertices = vertices @ rx.T
            vertices[:, 1] += 0.73 * charge * growth + 0.035 * math.sin(self.time * 2)
            screen = self._project(vertices, center, size)
            faces = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6),
                     (0, 3, 7, 4), (1, 2, 6, 5))
            for index in sorted(range(6), key=lambda i: vertices[list(faces[i]), 2].mean(), reverse=True):
                polygon = np.round(screen[list(faces[index])]).astype(np.int32)
                fill = tuple(int(c * (0.18 + 0.075 * index)) for c in color)
                cv2.fillConvexPoly(ink, polygon, fill, cv2.LINE_AA)
                cv2.fillConvexPoly(solid_mask, polygon, 255)
                cv2.polylines(ink, [polygon], True, tuple(int(c) for c in color), 2, cv2.LINE_AA)
            if self.armed or scene.particles:
                count = 28 if self.armed else 14
                for i in range(count):
                    phase = i * 2.399 + self.time * (2 if self.armed else 0.7)
                    point = (1.0 * math.cos(phase), 0.65 + 0.8 * math.sin(phase * 1.2), 1.0 * math.sin(phase))
                    x, y = self._project([point], center, size)[0]
                    self._star(ink, x, y, 2 + int(i % 4 == 0), color, 0.6 + 0.4 * math.sin(phase)**2)
        else:
            origin, burst_size = self._placement(self.burst_anchor, self.burst_span, self.burst_length, transform)
            # Frozen release position and scale: particles do not chase the hand.
            burst_size *= self.burst_scale
            origin[1] -= self.burst_height * burst_size * 0.5
            screen = self._project(self.positions, origin, burst_size)
            fade = max(0, 1 - self.burst_age / self.BURST_SECONDS)
            for i, (x, y) in enumerate(screen):
                twinkle = 0.3 + 0.7 * math.sin(self.time * (9 + i % 7) + self.twinkle[i])**2
                palette = (color, np.array((125, 220, 255)), np.array((255, 245, 228)))
                tint = palette[i % 3]
                radius = 1 + (i % 3)
                self._star(ink, x, y, radius, tint, fade * twinkle)
            radius = int(burst_size * (0.25 + self.burst_age * 1.4))
            if self.burst_age < 0.7:
                tint = tuple(int(c * (1 - self.burst_age / 0.7)) for c in color)
                cv2.ellipse(ink, tuple(origin.astype(int)), (max(2, radius), max(2, int(radius * 0.42))),
                            0, 0, 360, tint, 2, cv2.LINE_AA)

        glow = cv2.GaussianBlur(ink, (0, 0), 3.5)
        composed = cv2.addWeighted(base, 1.0, glow, 1.4, 0)
        composed = cv2.addWeighted(composed, 1.0, ink, 0.9, 0)
        # Sparkles add light; only cube surfaces darken/occlude the background.
        mask = solid_mask > 0
        composed[mask] = np.clip(base[mask].astype(float) * 0.30 + ink[mask] * 0.95 + glow[mask] * 0.5, 0, 255).astype(np.uint8)
        # Keep glow and particles out of letterbox bars and the surrounding UI.
        ox, oy, width, height = transform
        result = base.copy()
        result[oy:oy + height, ox:ox + width] = composed[oy:oy + height, ox:ox + width]
        return result

    @staticmethod
    def _star(image, x, y, radius, color, intensity):
        h, w = image.shape[:2]
        if not -10 <= x < w + 10 or not -10 <= y < h + 10:
            return
        point = (int(x), int(y))
        tint = tuple(int(c * intensity) for c in color)
        cv2.circle(image, point, radius, tint, -1, cv2.LINE_AA)
        if radius >= 3:
            cv2.line(image, (point[0] - 5, point[1]), (point[0] + 5, point[1]), tint, 1, cv2.LINE_AA)
            cv2.line(image, (point[0], point[1] - 5), (point[0], point[1] + 5), tint, 1, cv2.LINE_AA)
