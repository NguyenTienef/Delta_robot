"""Analytical position kinematics for a symmetric, rotary-input Delta robot.

The base lies in Z=0. Positive joint angles rotate the upper arms downwards.
The platform stays parallel to the base; only the lower assembly is used.
This module intentionally depends only on Python's standard library.
"""

from dataclasses import dataclass
import math
from typing import Iterable

Vector = tuple[float, float, float]


class KinematicsError(ValueError):
    """Unreachable position, joint limit, or singular configuration."""


def _add(a: Vector, b: Vector) -> Vector:
    return tuple(x + y for x, y in zip(a, b))


def _sub(a: Vector, b: Vector) -> Vector:
    return tuple(x - y for x, y in zip(a, b))


def _scale(a: Vector, s: float) -> Vector:
    return tuple(x * s for x in a)


def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: Vector, b: Vector) -> Vector:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _triplet(values: Iterable[float], name: str) -> Vector:
    try:
        result = tuple(float(x) for x in values)
    except (TypeError, ValueError) as exc:
        raise KinematicsError(f'{name}: can 3 gia tri so huu han.') from exc
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        raise KinematicsError(f'{name}: can 3 gia tri so huu han.')
    return result


@dataclass(frozen=True)
class Geometry:
    """Radii locate joint centers, rather than triangle vertices or side lengths."""

    base_radius: float = 150.0
    platform_radius: float = 45.0
    upper_arm: float = 150.0
    lower_arm: float = 330.0
    min_angle: float = -60.0
    max_angle: float = 90.0

    def __post_init__(self):
        lengths = (self.base_radius, self.platform_radius, self.upper_arm, self.lower_arm)
        if not all(math.isfinite(x) and x > 0 for x in lengths):
            raise ValueError('Kich thuoc robot phai duong va huu han.')
        if self.platform_radius >= self.base_radius:
            raise ValueError('Ban kinh ban dong phai nho hon ban kinh de.')
        if not (math.isfinite(self.min_angle) and math.isfinite(self.max_angle)
                and -180 <= self.min_angle < self.max_angle <= 180):
            raise ValueError('Gioi han goc khong hop le (-180 <= min < max <= 180).')


class DeltaRobot:
    def __init__(self, geometry: Geometry | None = None):
        self.geometry = geometry or Geometry()
        self.directions = tuple((math.cos(math.radians(a)), math.sin(math.radians(a)), 0.0)
                                for a in (-90, 30, 150))

    def _angles(self, angles: Iterable[float]) -> Vector:
        result = _triplet(angles, 'Goc')
        g = self.geometry
        if any(a < g.min_angle - 1e-8 or a > g.max_angle + 1e-8 for a in result):
            raise KinematicsError(f'Vuot gioi han goc [{g.min_angle:g}, {g.max_angle:g}] do.')
        return result

    def _elbows(self, angles: Vector) -> tuple[Vector, ...]:
        g = self.geometry
        return tuple(_add(_scale(u, g.base_radius + g.upper_arm * math.cos(math.radians(a))),
                          (0, 0, -g.upper_arm * math.sin(math.radians(a))))
                     for u, a in zip(self.directions, angles))

    def forward(self, angles: Iterable[float]) -> Vector:
        """Three motor angles (degrees) -> platform center (mm).

        Translate each elbow by the corresponding platform attachment offset.
        The platform center is the lower intersection of three equal spheres.
        An orthonormal basis avoids dividing by Cartesian Z or assuming equal angles.
        """
        g = self.geometry
        elbows = self._elbows(self._angles(angles))
        c1, c2, c3 = tuple(_sub(e, _scale(u, g.platform_radius))
                           for e, u in zip(elbows, self.directions))
        delta = _sub(c2, c1)
        distance = math.sqrt(_dot(delta, delta))
        scale = max(g.base_radius, g.upper_arm, g.lower_arm)
        if distance < 1e-10 * scale:
            raise KinematicsError('Cau hinh ky di: tam cac mat cau trung nhau.')
        ex = _scale(delta, 1 / distance)
        offset = _sub(c3, c1)
        i = _dot(ex, offset)
        perpendicular = _sub(offset, _scale(ex, i))
        j = math.sqrt(_dot(perpendicular, perpendicular))
        if j < 1e-10 * scale:
            raise KinematicsError('Cau hinh ky di: tam cac mat cau thang hang.')
        ey = _scale(perpendicular, 1 / j)
        ez = _cross(ex, ey)
        x = distance / 2
        y = (i * i + j * j - 2 * i * x) / (2 * j)
        height_sq = g.lower_arm**2 - x**2 - y**2
        if height_sq < -1e-10 * scale**2:
            raise KinematicsError('Ba canh tay khong co giao diem thuc.')
        if height_sq <= 1e-12 * scale**2:
            raise KinematicsError('Cau hinh ky di: hai nghiem FK tiep xuc.')
        center = _add(c1, _add(_scale(ex, x), _scale(ey, y)))
        height = math.sqrt(height_sq)
        solutions = (_add(center, _scale(ez, height)), _add(center, _scale(ez, -height)))
        result = min(solutions, key=lambda p: p[2])
        if result[2] >= -1e-9:
            raise KinematicsError('Ban dong phai nam ben duoi mat de (Z < 0).')
        return result

    def inverse(self, position: Iterable[float]) -> Vector:
        """XYZ (mm) -> outer-elbow motor angles (degrees).

        For each radial direction u: A*cos(theta) + B*sin(theta) = C,
        where d = P + (Rp-Rb)*u, A=d.u, B=-Pz,
        C=(d.d + L^2 - l^2)/(2L). Select phase-acos(C/hypot(A,B)).
        """
        p = _triplet(position, 'Vi tri')
        if p[2] >= -1e-9:
            raise KinematicsError('Ban dong phai nam ben duoi mat de (Z < 0).')
        g = self.geometry
        angles = []
        for index, u in enumerate(self.directions, start=1):
            d = _add(p, _scale(u, g.platform_radius - g.base_radius))
            a, b = _dot(d, u), -p[2]
            c = (_dot(d, d) + g.upper_arm**2 - g.lower_arm**2) / (2 * g.upper_arm)
            amplitude = math.hypot(a, b)
            if amplitude < 1e-10 or abs(c) > amplitude * (1 + 1e-12):
                raise KinematicsError(f'Ngoai vung lam viec: canh tay {index}.')
            angle = math.atan2(b, a) - math.acos(max(-1.0, min(1.0, c / amplitude)))
            angles.append(math.degrees(angle))
        result = self._angles(angles)
        # Reject targets belonging to another assembly mode.
        if math.dist(self.forward(result), p) > 1e-7 * max(1, g.lower_arm):
            raise KinematicsError('Vi tri khong thuoc nhanh lap rap phia duoi.')
        return result

    def joints(self, angles: Iterable[float], position: Iterable[float] | None = None):
        """Return the three base joints, elbows and platform joints for drawing."""
        a = self._angles(angles)
        p = self.forward(a) if position is None else _triplet(position, 'Vi tri')
        g = self.geometry
        base = tuple(_scale(u, g.base_radius) for u in self.directions)
        platform = tuple(_add(p, _scale(u, g.platform_radius)) for u in self.directions)
        return base, self._elbows(a), platform
