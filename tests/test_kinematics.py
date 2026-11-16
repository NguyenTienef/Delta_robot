import math
import random
import unittest

from delta_sim.kinematics import DeltaRobot, Geometry, KinematicsError


class KinematicsTests(unittest.TestCase):
    def setUp(self):
        self.robot = DeltaRobot()

    def test_horizontal_arms_have_analytical_center_position(self):
        # At zero degrees: radial separation = 150 + 150 - 45 = 255 mm.
        expected_z = -math.sqrt(330**2 - 255**2)
        x, y, z = self.robot.forward((0, 0, 0))
        self.assertAlmostEqual(x, 0, places=9)
        self.assertAlmostEqual(y, 0, places=9)
        self.assertAlmostEqual(z, expected_z, places=9)
        for angle in self.robot.inverse((0, 0, expected_z)):
            self.assertAlmostEqual(angle, 0, places=9)

    def test_positive_angles_lower_platform(self):
        expected_z = -75 - math.sqrt(330**2 - (105 + 75 * math.sqrt(3))**2)
        self.assertAlmostEqual(self.robot.forward((30, 30, 30))[2], expected_z, places=9)

    def test_cartesian_round_trip_and_independent_link_lengths(self):
        for target in ((0, 0, -300), (65, -30, -280), (-70, 40, -320),
                       (50, 60, -250), (-20, -65, -350)):
            with self.subTest(target=target):
                angles = self.robot.inverse(target)
                actual = self.robot.forward(angles)
                self.assertLess(math.dist(actual, target), 1e-8)
                base, elbows, platform = self.robot.joints(angles, actual)
                for b, e, p in zip(base, elbows, platform):
                    self.assertAlmostEqual(math.dist(b, e), 150, places=8)
                    self.assertAlmostEqual(math.dist(e, p), 330, places=8)
                    self.assertAlmostEqual(math.dist(p, actual), 45, places=8)

    def test_asymmetric_joint_round_trip(self):
        rng = random.Random(42)
        for _ in range(150):
            angles = tuple(rng.uniform(5, 65) for _ in range(3))
            recovered = self.robot.inverse(self.robot.forward(angles))
            for actual, expected in zip(recovered, angles):
                self.assertAlmostEqual(actual, expected, places=7)

    def test_unreachable_and_nonfinite_targets_are_rejected(self):
        for target in ((1000, 0, -300), (0, 0, 0), (0, 0, 100),
                       (0, 0, -1000), (math.nan, 0, -300), (0, math.inf, -300)):
            with self.subTest(target=target), self.assertRaises(KinematicsError):
                self.robot.inverse(target)

    def test_joint_limits_and_bad_triplets_are_rejected(self):
        for angles in ((0, 0, 120), (0, -90, 0), (math.nan, 0, 0), (0, 0)):
            with self.subTest(angles=angles), self.assertRaises(KinematicsError):
                self.robot.forward(angles)

    def test_degenerate_sphere_centers_are_rejected(self):
        robot = DeltaRobot(Geometry(base_radius=100, platform_radius=50,
                                   upper_arm=100, min_angle=-180, max_angle=180))
        with self.assertRaises(KinematicsError):
            robot.forward((120, 120, 120))

    def test_invalid_geometry_is_rejected(self):
        for kwargs in ({'upper_arm': 0}, {'lower_arm': -1}, {'base_radius': math.nan},
                       {'platform_radius': 160}, {'min_angle': 90, 'max_angle': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Geometry(**kwargs)


if __name__ == '__main__':
    unittest.main()
