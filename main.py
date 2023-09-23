from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
import math
from functools import cached_property

import pygame
from pygame.locals import *

from math import inf

RESOLUTION = (1280, 720)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
POINT_SIZE = 2

Point = Vector = pygame.Vector2

import sys

sys.setrecursionlimit(1000)

@dataclass
class Ray:
    origin: Point
    direction: Vector
    reflected: bool = False

    def draw(self, surface: pygame.Surface):
        VERY_BIG = 2*max(RESOLUTION)

        pygame.draw.line(surface, (255, 0, 0) if self.reflected else GRAY, self.origin, self.origin+self.direction.normalize()*VERY_BIG)
        draw_point(surface, self.origin)


@dataclass
class Interface:
    start: Point
    end: Point
    front_indice: float = 1.0
    back_indice: float = 1.0

    @cached_property
    def director(self) -> Vector:
        return (self.end - self.start).normalize()

    @cached_property
    def normal(self) -> Vector:
        # normal is pointing up when start is to the left of end
        #    ↑n
        # A---->B
        return Vector(-self.director.y, self.director.x)

    @cached_property
    def normal_angle(self) -> float:
        return self.normal.as_polar()[1]

    def refract(self, direction: Vector, at: Point) -> Optional[Ray]:
        dir_amp, dir_angle = direction.as_polar()

        i1 = self.normal_angle - dir_angle

        refracted_ray = Vector.from_polar((1.0, 180+2*self.normal_angle-dir_angle))

        return Ray(origin=at, direction=refracted_ray, reflected=False)

    def draw(self, surface: pygame.Surface):
        pygame.draw.line(surface, GRAY, self.start, self.end)

        draw_point(surface, self.start, (0, 255, 0))
        draw_point(surface, self.end, (255, 0, 255))


class RayCaster:
    def __init__(self, interfaces: list[Interface]):
        self.interfaces = interfaces

    def project(self, *rays: Ray, limit: int = sys.getrecursionlimit()-100, ignore: int = -1) -> list[Ray]:
        if limit == 0:
            return list(rays)

        o = rays[-1].origin
        r = rays[-1].direction

        intersec_t = math.inf
        intersected_interface = None
        for index, interf in enumerate(self.interfaces):
            if index == ignore:
                continue

            a = interf.start
            b = interf.end

            determinant = (a.y - b.y) * r.x - (a.x - b.x) * r.y

            if determinant == 0:
                continue

            t = (a.y - b.y) * (a.x - o.x) + (a.y - o.y) * (b.x - a.x)
            s = (a.y - o.y) * r.x - (a.x - o.x) * r.y

            t /= determinant
            s /= determinant

            if not (0 < s < 1 and t > 0):
                continue

            if t < intersec_t:
                intersec_t = t
                intersected_interface = index

        if math.isinf(intersec_t):
            return list(rays)

        refracted = self.interfaces[intersected_interface].refract(
            direction=rays[-1].direction,
            at=Point(
                x=o.x + r.x * intersec_t,
                y=o.y + r.y * intersec_t
            )
        )

        if refracted is None:
            return list(rays)

        return self.project(
            *rays,
            refracted,
            limit=limit-1,
            ignore=intersected_interface
        )

    def draw(self, surface: pygame.Surface):
        for interf in self.interfaces:
            interf.draw(surface)

    def dump_interfaces(self):
        print("BEGIN INTERFACE DUMP")
        for interf in self.interfaces:
            print(f"Interface(start=Point({interf.start.x}, {interf.start.y}), " 
                  f"end=Point({interf.end.x}, {interf.end.y}), "
                  f"front_indice={interf.front_indice}, back_indice={interf.back_indice}),")
        print("END INTERFACE DUMP")

def draw_point(surface: pygame.Surface, point: Point, color: (int, int, int) = WHITE):
    pygame.draw.circle(surface, color, point, POINT_SIZE)


def draw_segment(surface: pygame.Surface, ray: Ray, end: Point):
    pygame.draw.line(surface, (255, 0, 0) if ray.reflected else GRAY, ray.origin, end)

    draw_point(surface, ray.origin)
    draw_point(surface, end)

screen = pygame.display.set_mode(RESOLUTION)
clock = pygame.time.Clock()

ray = Ray(
    origin=Point(RESOLUTION[0].__float__(), RESOLUTION[1]/2),
    direction=Vector(-1.0, 0.0),
)

raycaster = RayCaster([
])

t = 0.0
paused = False
running = True
point_buffer = None
ray_start = None

while running:
    t += clock.tick(60) / 1000 * (not paused)

    for ev in pygame.event.get():
        if ev.type == QUIT:
            running = False
            break
        if ev.type == KEYUP:
            if ev.key == K_SPACE:
                paused = not paused
            if ev.key == K_d:
                raycaster.dump_interfaces()
            if ev.key == K_c:
                if point_buffer is not None:
                    point_buffer = None
                if len(raycaster.interfaces) != 0:
                    raycaster.interfaces.pop()

        if ev.type == MOUSEBUTTONDOWN:
            if ev.button == 1:
                if point_buffer is None:
                    point_buffer = ev.pos
                else:
                    raycaster.interfaces.append(Interface(
                        Point(*point_buffer),
                        Point(*ev.pos),
                        front_indice=inf
                    ))
                    point_buffer = None
            if ev.button == 3:
                ray_start = ev.pos

        if ev.type == MOUSEBUTTONUP:
            if ev.button == 3:
                diff = Vector(ev.pos)-Vector(ray_start)
                if diff != Vector(0.0, 0.0):
                    ray = Ray(origin=Vector(ray_start), direction=diff)


    screen.fill(BLACK)

    refracted = raycaster.project(ray)

    raycaster.draw(screen)

    if len(refracted) == 1:
        refracted[0].draw(screen)
    else:
        next_ray = None
        for refracted_ray, next_ray in zip(refracted[:-1], refracted[1:]):
            draw_segment(screen, refracted_ray, next_ray.origin)

        next_ray.draw(screen)

    if point_buffer is not None:
        draw_point(screen, point_buffer)

    pygame.display.flip()
