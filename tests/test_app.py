import os
os.environ['MPLBACKEND'] = 'Agg'

import math
from pathlib import Path
import tempfile
import unittest

from delta_sim.app import Simulator, MODE_FK, MODE_IK
from delta_sim.kinematics import DeltaRobot
from delta_sim.trajectory import make_path, solve_path


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.robot = DeltaRobot()
        self.app = Simulator(self.robot, solve_path(self.robot, make_path(samples=41)))

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close(self.app.fig)

    def test_playback_moves_robot_and_draws_traversed_path(self):
        self.app.render_frame(10)
        self.assertLess(math.dist(self.app.pose.actual, (0, 70, -300)), 1e-8)
        x, y, z = self.app.trace.get_data_3d()
        self.assertEqual(len(x), 11)
        self.assertAlmostEqual(y[-1], 70)
        self.app.fig.canvas.draw()

    def test_fk_sliders_control_motor_angles(self):
        self.app.set_mode(MODE_FK)
        for slider in self.app.sliders:
            slider.set_val(30)
        expected_z = -75 - math.sqrt(330**2 - (105 + 75 * math.sqrt(3))**2)
        self.assertEqual(self.app.pose.angles, (30, 30, 30))
        self.assertAlmostEqual(self.app.pose.actual[2], expected_z, places=8)

    def test_invalid_ik_keeps_last_valid_robot_pose(self):
        self.app.set_mode(MODE_IK)
        self.app.sliders[0].set_val(20)
        self.assertAlmostEqual(self.app.pose.actual[0], 20, places=8)
        previous = self.app.pose
        self.app.sliders[2].set_val(-30)
        self.assertEqual(self.app.pose, previous)
        self.assertIn('Khong hop le', self.app.status.get_text())

    def test_restart_resumes_paused_playback(self):
        self.app._toggle_pause()
        self.app.elapsed = 3
        self.app.render_frame(10)
        self.app._reset()
        self.assertEqual(self.app.index, 0)
        self.assertEqual(self.app.elapsed, 0)
        self.assertFalse(self.app.paused)

    def test_single_pose_csv_can_switch_to_a_sample_trajectory(self):
        self.app.poses = [self.app.pose]
        self.app.path_name = 'csv'
        self.app._select_path('Số 8')
        self.assertGreaterEqual(len(self.app.poses), 5)
        self.assertEqual(self.app.path_name, 'figure8')

    def test_restart_restores_last_valid_path_controls(self):
        self.app.sliders[1].set_val(-180)
        self.assertTrue(self.app._invalid_controls)
        self.app._reset()
        self.assertFalse(self.app._invalid_controls)
        self.assertEqual(self.app.sliders[1].val, -300)

    def test_failed_path_selection_restores_selected_radio(self):
        self.app.sliders[1].set_val(-350)
        self.app.sliders[0].set_val(120)
        self.app.path_radio.set_active(2)
        self.assertEqual(self.app.path_name, 'circle')
        self.assertEqual(self.app.path_radio.value_selected, 'Tròn')

    def test_gif_preserves_duration_at_30fps(self):
        from PIL import Image
        self.app.duration = 1
        self.app.fps = 30
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.gif'
            self.app.save_gif(path)
            with Image.open(path) as image:
                duration_ms = 0
                for index in range(image.n_frames):
                    image.seek(index)
                    duration_ms += image.info['duration']
                self.assertEqual(duration_ms, 1000)


if __name__ == '__main__':
    unittest.main()
