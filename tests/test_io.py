import csv
from pathlib import Path
import tempfile
import unittest

from delta_sim.kinematics import DeltaRobot
from delta_sim.trajectory import make_path, read_csv, solve_path, write_csv


class CsvTests(unittest.TestCase):
    def test_export_can_be_replayed_with_angles_and_timestamps(self):
        poses = solve_path(DeltaRobot(), make_path(samples=9))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion.csv'
            write_csv(path, poses, duration=4)
            with path.open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(float(rows[0]['t_s']), 0)
            self.assertEqual(float(rows[-1]['t_s']), 4)
            self.assertAlmostEqual(float(rows[3]['theta2_deg']), poses[3].angles[1], places=7)
            for actual, expected in zip(read_csv(path), [p.target for p in poses]):
                for a, b in zip(actual, expected):
                    self.assertAlmostEqual(a, b, places=7)

    def test_bad_csv_data_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            for text in ('x,y,z\n0,0,-300\n', 'x_mm,y_mm,z_mm\n',
                         'x_mm,y_mm,z_mm\n0,nan,-300\n',
                         'x_mm,y_mm,z_mm\n0,no,-300\n'):
                path.write_text(text, encoding='utf-8')
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_csv(path)


if __name__ == '__main__':
    unittest.main()
