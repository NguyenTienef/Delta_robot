import math
import unittest

from delta_sim.kinematics import DeltaRobot, KinematicsError
from delta_sim.trajectory import PATH_NAMES, make_path, solve_path


class TrajectoryTests(unittest.TestCase):
    def test_all_presets_are_closed_and_reachable(self):
        for name in PATH_NAMES:
            with self.subTest(name=name):
                points = make_path(name, samples=121)
                self.assertEqual(len(points), 121)
                self.assertLess(math.dist(points[0], points[-1]), 1e-9)
                poses = solve_path(DeltaRobot(), points)
                self.assertEqual(len(poses), 121)
                self.assertLess(max(p.error for p in poses), 1e-8)

    def test_circle_radius_and_plane(self):
        for x, y, z in make_path('circle', radius=60, z=-280):
            self.assertAlmostEqual(math.hypot(x, y), 60, places=9)
            self.assertEqual(z, -280)

    def test_wave_moves_in_three_dimensions(self):
        points = make_path('wave')
        self.assertGreater(max(p[2] for p in points) - min(p[2] for p in points), 40)

    def test_invalid_parameters_and_paths_are_rejected(self):
        for kwargs in ({'name': 'unknown'}, {'radius': -2}, {'samples': 1},
                       {'radius': math.nan}, {'z': 0}, {'samples': 3.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_path(**kwargs)
        with self.assertRaises(KinematicsError):
            solve_path(DeltaRobot(), [(0, 0, -300), (1000, 0, -300)])
        with self.assertRaises(ValueError):
            solve_path(DeltaRobot(), [])


if __name__ == '__main__':
    unittest.main()
