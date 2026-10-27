"""A real perspective-projected 3D cube, floor and particle field (CPU)."""
import math
import numpy as np


class Scene3D:
    def __init__(self):
        self.angle = 0.35
        self.running = False
        self.scale = 1.0
        self.height = 0.0
        self.color_index = 0
        self.particles = False
        self.time = 0.0
        self.pulse = 0.0
        self.colors = ((73, 222, 205), (144, 116, 255), (255, 174, 91), (101, 182, 255))
        self.vertices = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                                  [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=float)

    def apply(self, command):
        if command == "START":
            self.running = True
        elif command == "STOP":
            self.running = False
        elif command == "MOVE_UP":
            self.height = min(self.height + 0.3, 0.9)
        elif command == "ZOOM_IN":
            self.scale = min(1.5, self.scale + 0.15)
        elif command == "ZOOM_OUT":
            self.scale = max(0.45, self.scale - 0.15)
        elif command == "CHANGE_COLOR":
            self.color_index = (self.color_index + 1) % len(self.colors)
        elif command == "PARTICLES":
            self.particles = not self.particles
        else:
            return
        self.pulse = 1.0

    def update(self, dt):
        dt = min(max(dt, 0), 0.1)
        self.time += dt
        if self.running:
            self.angle += dt * 0.9
        self.pulse = max(0, self.pulse - dt * 1.5)

    @staticmethod
    def project(points, center, focal=290):
        points = np.asarray(points, dtype=float)
        depth = points[:, 2] + 7.5
        return np.column_stack((center[0] + points[:, 0] * focal / depth,
                                center[1] - points[:, 1] * focal / depth))

    def draw(self, draw, rect):
        x0, y0, x1, y1 = rect
        center = ((x0 + x1) / 2, (y0 + y1) / 2 + 28)
        color = self.colors[self.color_index]
        # Perspective floor, contained inside the scene viewport.
        for offset in np.linspace(-3, 3, 9):
            for points in ([(offset, -1.65, -2.5), (offset, -1.65, 3.5)],
                           [(-3, -1.65, offset), (3, -1.65, offset)]):
                projected = self.project(points, center)
                draw.line([tuple(p) for p in projected], fill="#24384c", width=1)
        angle = self.angle
        ry = np.array([[math.cos(angle), 0, math.sin(angle)], [0,1,0], [-math.sin(angle),0,math.cos(angle)]])
        tilt = -0.32
        rx = np.array([[1,0,0], [0,math.cos(tilt),-math.sin(tilt)], [0,math.sin(tilt),math.cos(tilt)]])
        world = (self.vertices * self.scale) @ ry.T @ rx.T
        world[:, 1] += self.height
        screen = self.project(world, center)
        faces = ((0,1,2,3), (4,5,6,7), (0,1,5,4), (2,3,7,6), (0,3,7,4), (1,2,6,5))
        # Painter's algorithm: far faces first, with distinct face shading.
        for index in sorted(range(6), key=lambda i: float(world[list(faces[i]), 2].mean()), reverse=True):
            face = faces[index]
            shade = 0.24 + 0.085 * index
            fill = tuple(int(c * shade) for c in color)
            polygon = [tuple(screen[i]) for i in face]
            draw.polygon(polygon, fill=fill)
            draw.line(polygon + [polygon[0]], fill=color, width=2)
        if self.particles:
            for i in range(48):
                phase = i * 2.399 + self.time * 0.6
                radius = 1.7 + 0.4 * math.sin(i * 3 + self.time)
                point = (radius * math.cos(phase), 1.2 * math.sin(phase * 1.4) + self.height, radius * math.sin(phase))
                x, y = self.project([point], center)[0]
                size = 2 if point[2] < 0 else 3
                draw.ellipse((x-size, y-size, x+size, y+size), fill=color)
        if self.pulse:
            radius = 65 + 40 * (1 - self.pulse)
            draw.ellipse((center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius), outline=color, width=2)
