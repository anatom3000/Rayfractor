from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from math import sin

import pygame
from pygame.locals import *

RESOLUTION = (640, 480)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)

Point = namedtuple("Point", ['x', 'y'])
Vector = namedtuple("Vector", ['x', 'y'])


@dataclass
class Ray:
    origin: Point
    direction: Vector

class RayCaster:
    def __init__(self, segment: (Point, Point)):
        self.segment = segment

    def raycast(self, ray: Ray) -> Optional[Point]:
        a = self.segment[0]
        b = self.segment[1]
        o = ray.origin
        r = ray.direction

        determinant = (a.y - b.y)*r.x - (a.x - a.x)*r.y

        if determinant == 0:
            return None

        t = (a.y - b.y)*(a.x - o.x) + (a.y - o.y)*(b.x-a.x)
        s = (o.x - a.x)*r.y + (a.y - o.y)*r.x

        t /= determinant
        s /= determinant

        if not (0 < s < 1 and t > 0):
            return None

        return Point(
            x=o.x + r.x * t,
            y=o.y + r.y * t
        )


def draw_point(surface: pygame.Surface, point: Point):
    pygame.draw.circle(surface, WHITE, (point.x, point.y), 5)


def draw_segment(surface: pygame.Surface, segment: (Point, Point)):
    pygame.draw.line(surface, GRAY, segment[0], segment[1])

    draw_point(surface, segment[0])
    draw_point(surface, segment[1])


screen = pygame.display.set_mode(RESOLUTION)
clock = pygame.time.Clock()

raycaster = RayCaster(segment=(Point(400.0, 100.0), Point(400.0, 400.0)))

t = 0.0
running = True
while running:
    t += clock.tick(60) / 1000

    for ev in pygame.event.get():
        if ev.type == QUIT:
            running = False
            break

    screen.fill(BLACK)

    ray = Ray(
        origin=Point(100.0, 250.0),
        direction=Vector(50.0, 30.0*sin(t))
    )
    casted = raycaster.raycast(ray)

    draw_segment(screen, raycaster.segment)

    if casted is None:
        draw_point(screen, ray.origin)
    else:
        draw_segment(screen, (ray.origin, casted))

    pygame.display.flip()
