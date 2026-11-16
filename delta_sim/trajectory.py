"""Closed sample paths and preflight validation, independent of the GUI."""

from dataclasses import dataclass
import csv
import math
from pathlib import Path

from .kinematics import DeltaRobot, KinematicsError, Vector

PATH_NAMES = ('circle', 'figure8', 'square', 'wave')


@dataclass(frozen=True)
class Pose:
    target: Vector
    actual: Vector
    angles: Vector

    @property
    def error(self) -> float:
        return math.dist(self.target, self.actual)


def make_path(name='circle', radius=70.0, z=-300.0, samples=241) -> list[Vector]:
    if name not in PATH_NAMES:
        raise ValueError(f'Quy dao khong hop le: {name}.')
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Ban kinh phai duong va huu han.')
    if not math.isfinite(z) or z >= 0:
        raise ValueError('Z phai am va huu han.')
    if not isinstance(samples, int) or samples < 5:
        raise ValueError('So mau phai la so nguyen >= 5.')
    points = []
    corners = ((radius, radius), (-radius, radius), (-radius, -radius), (radius, -radius))
    for index in range(samples):
        fraction = index / (samples - 1)
        t = 2 * math.pi * fraction
        if name == 'figure8':
            point = (radius * math.sin(t), radius * math.sin(2 * t) / 2, z)
        elif name == 'square':
            segment = min(int(fraction * 4), 3)
            weight = fraction * 4 - segment
            a, b = corners[segment], corners[(segment + 1) % 4]
            point = (a[0] + weight * (b[0] - a[0]), a[1] + weight * (b[1] - a[1]), z)
        else:
            point = (radius * math.cos(t), radius * math.sin(t),
                     z + 25 * math.sin(2 * t) if name == 'wave' else z)
        points.append(point)
    points[-1] = points[0]
    return points


def solve_path(robot: DeltaRobot, points) -> list[Pose]:
    poses = []
    for index, point in enumerate(points):
        try:
            angles = robot.inverse(point)
            poses.append(Pose(tuple(point), robot.forward(angles), angles))
        except KinematicsError as exc:
            raise KinematicsError(f'Mau {index + 1}: {exc}') from exc
    if not poses:
        raise ValueError('Quy dao phai co it nhat mot diem.')
    return poses


def write_csv(path, poses: list[Pose], duration: float = 8.0):
    """Export uniformly spaced samples; XYZ columns can be imported again."""
    if not poses or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Can cac mau hop le va thoi gian duong.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('t_s', 'x_mm', 'y_mm', 'z_mm', 'theta1_deg', 'theta2_deg',
                         'theta3_deg', 'fk_x_mm', 'fk_y_mm', 'fk_z_mm', 'error_mm'))
        for index, pose in enumerate(poses):
            t = duration * index / max(1, len(poses) - 1)
            writer.writerow((t, *pose.target, *pose.angles, *pose.actual, pose.error))


def read_csv(path) -> list[Vector]:
    """Read XYZ samples in row order, rescheduled uniformly by the simulator.

    An exported t_s column is informational; --duration controls replay timing.
    The full imported path is validated by solve_path before it is displayed.
    """
    points = []
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'x_mm', 'y_mm', 'z_mm'}.issubset(reader.fieldnames or ()):
            raise ValueError('CSV can cac cot x_mm, y_mm, z_mm.')
        for index, row in enumerate(reader, start=2):
            try:
                point = tuple(float(row[key]) for key in ('x_mm', 'y_mm', 'z_mm'))
                if not all(math.isfinite(x) for x in point):
                    raise ValueError('nonfinite')
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Du lieu CSV khong hop le tai dong {index}.') from exc
            points.append(point)
    if not points:
        raise ValueError('CSV khong co diem quy dao.')
    return points
