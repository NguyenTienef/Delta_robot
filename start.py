"""Run the Delta robot desktop simulator or generate headless artifacts."""

import argparse
import math
import os
from pathlib import Path
import sys

from delta_sim import DeltaRobot, Geometry
from delta_sim.trajectory import PATH_NAMES, make_path, read_csv, solve_path, write_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description='Delta robot: FK / IK / 3D trajectory simulation')
    parser.add_argument('--path', choices=PATH_NAMES, default='circle')
    parser.add_argument('--radius', type=float, default=70, help='Path radius / square half-side (mm)')
    parser.add_argument('--z', type=float, default=-300, help='Path height, below base (mm)')
    parser.add_argument('--samples', type=int, default=241)
    parser.add_argument('--duration', type=float, default=8, help='One playback cycle in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Requested display / GIF frame rate')
    parser.add_argument('--base-radius', type=float, default=150)
    parser.add_argument('--platform-radius', type=float, default=45)
    parser.add_argument('--upper-arm', type=float, default=150)
    parser.add_argument('--lower-arm', type=float, default=330)
    parser.add_argument('--replay', type=Path, help='CSV containing x_mm,y_mm,z_mm columns')
    parser.add_argument('--export-csv', type=Path)
    parser.add_argument('--preview', type=Path, help='Save a PNG without opening a window')
    parser.add_argument('--gif', type=Path, help='Save a complete animation as GIF')
    parser.add_argument('--check', action='store_true', help='Validate motion without graphics libraries')
    parser.add_argument('--show', action='store_true', help='Also open GUI after generating outputs')
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.duration) or args.duration <= 0:
            raise ValueError('Duration must be finite and positive.')
        if not 1 <= args.fps <= 120:
            raise ValueError('FPS must be between 1 and 120.')
        robot = DeltaRobot(Geometry(base_radius=args.base_radius,
                                   platform_radius=args.platform_radius,
                                   upper_arm=args.upper_arm, lower_arm=args.lower_arm))
        points = read_csv(args.replay) if args.replay else make_path(
            args.path, args.radius, args.z, args.samples)
        poses = solve_path(robot, points)
        print(f'{len(poses)} samples | max IK/FK error: {max(p.error for p in poses):.3e} mm')
        if args.export_csv:
            write_csv(args.export_csv, poses, args.duration)
            print(f'CSV: {args.export_csv}')
        outputs_only = args.check or args.export_csv or args.preview or args.gif
        if not (args.show or args.preview or args.gif or not outputs_only):
            return 0
        os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.cache' / 'matplotlib'))
        try:
            import matplotlib
            if outputs_only and not args.show:
                matplotlib.use('Agg')
            from delta_sim.app import Simulator
        except ImportError as exc:
            print(f'Missing graphics dependency: {exc}\n'
                  'Run: .venv\\Scripts\\python.exe -m pip install -r requirements.txt\n'
                  'Or run run.bat using the prepared environment.', file=sys.stderr)
            return 1
        app = Simulator(robot, poses, duration=args.duration, fps=args.fps,
                        path_name='csv' if args.replay else args.path, radius=args.radius, z=args.z)
        if args.preview:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            app.render_frame(len(poses) // 3)
            app.fig.savefig(args.preview, dpi=140, facecolor=app.fig.get_facecolor())
            print(f'Preview: {args.preview}')
        if args.gif:
            app.save_gif(args.gif)
            print(f'GIF: {args.gif}')
        if args.show or not outputs_only:
            app.render_frame(0)
            app.show()
        else:
            import matplotlib.pyplot as plt
            plt.close(app.fig)
        return 0
    except (ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
