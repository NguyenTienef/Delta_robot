"""Matplotlib desktop UI: trajectory playback, Cartesian IK and joint-space FK."""

import math
import os
from pathlib import Path
import sys
import time
import textwrap

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image

from .kinematics import DeltaRobot, KinematicsError
from .trajectory import PATH_NAMES, Pose, make_path, solve_path, write_csv

# A portable Windows environment can carry its matching Tcl/Tk scripts locally.
# Prefer these when present; normal Python installations keep their own defaults.
for _variable, _folder in (('TCL_LIBRARY', 'tcl8.6'), ('TK_LIBRARY', 'tk8.6')):
    _library = Path(sys.prefix) / 'tcl' / _folder
    if _library.is_dir():
        os.environ.setdefault(_variable, str(_library))

MODE_PATH = 'Quỹ đạo'
MODE_IK = 'IK / XYZ'
MODE_FK = 'FK / Góc'
PATH_LABELS = ('Tròn', 'Số 8', 'Vuông', 'Sóng 3D')
BACKGROUND = '#0d1524'
PANEL = '#131f32'
TEXT = '#e3ecf7'
MUTED = '#97abc4'
ACCENT = '#53ddc6'
COLORS = ('#53ddc6', '#ffb963', '#a398ff')


class Simulator:
    def __init__(self, robot: DeltaRobot, poses: list[Pose], duration=8.0,
                 path_name='circle', radius=70.0, z=-300.0, fps=30):
        self.robot = robot
        self.poses = poses
        self.duration, self.fps = duration, fps
        self.path_name, self.radius, self.z = path_name, radius, z
        self.speed = 1.0
        self.mode = MODE_PATH
        self.pose = poses[0]
        self.index = 0
        self.elapsed = 0.0
        self.paused = False
        self.last_tick = time.monotonic()
        self.animation = None
        self.sliders = []
        self._invalid_controls = False
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'text.color': TEXT, 'axes.labelcolor': MUTED,
                             'xtick.color': MUTED, 'ytick.color': MUTED,
                             'axes.edgecolor': '#2b405b'}):
            self.fig = plt.figure(figsize=(15, 9), facecolor=BACKGROUND)
            self.fig.canvas.manager.set_window_title('Delta Robot | Mô phỏng động học')
            self._build_scene()
            self._build_controls()
        self._refresh_path()
        self.render_frame(0)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _build_scene(self):
        self.fig.text(.04, .946, 'DELTA ROBOT', fontsize=25, weight='bold', color=TEXT)
        self.fig.text(.04, .908, 'MÔ PHỎNG ĐỘNG HỌC  /  3 BẬC TỰ DO', fontsize=10, color=ACCENT)
        self.fig.text(.04, .035, 'Kéo chuột để xoay 3D  ·  Space: dừng / chạy  ·  R: chạy lại',
                      fontsize=10, color=MUTED)
        g = self.robot.geometry
        self.fig.text(.04, .075,
                      f'R đế {g.base_radius:g}   /   R bàn động {g.platform_radius:g}   /   '
                      f'Tay trên {g.upper_arm:g}   /   Tay dưới {g.lower_arm:g} mm', color=MUTED)
        self.ax = self.fig.add_axes((.015, .15, .60, .73), projection='3d', facecolor=BACKGROUND)
        extent = g.base_radius + g.upper_arm + 30
        depth = g.upper_arm + g.lower_arm + 20
        self.ax.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-depth, 40),
                    xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
        self.ax.set_box_aspect((2 * extent, 2 * extent, depth + 40))
        self.ax.view_init(elev=20, azim=38)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((.07, .11, .18, 1))
            axis.set_tick_params(colors=MUTED, labelsize=8)
        self.ax.grid(alpha=.15)
        base = np.array([np.array(u) * g.base_radius for u in self.robot.directions])
        self.ax.add_collection3d(Poly3DCollection([base], facecolors='#304561',
                                                 edgecolors='#6c849e', alpha=.40))
        t = np.linspace(0, 2 * np.pi, 120)
        self.ax.plot(g.base_radius * np.cos(t), g.base_radius * np.sin(t),
                     np.zeros_like(t), color='#6c849e', alpha=.5, lw=1)
        self.upper_lines, self.lower_lines = [], []
        for i, color in enumerate(COLORS):
            self.ax.plot(*base[i], marker='o', color=color, markersize=8)
            self.upper_lines.append(self.ax.plot([], [], [], color=color, lw=5,
                                                 marker='o', markersize=5)[0])
            self.lower_lines.append([self.ax.plot([], [], [], color=color, lw=2.1, alpha=.85)[0]
                                     for _ in range(2)])
            self.ax.text(*base[i], f'  M{i + 1}', color=color, fontsize=9)
        self.platform = Poly3DCollection([np.zeros((3, 3))], facecolors=ACCENT,
                                         edgecolors=ACCENT, alpha=.3)
        self.ax.add_collection3d(self.platform)
        self.tool, = self.ax.plot([], [], [], 'o', color='white', markersize=6)
        self.path_line, = self.ax.plot([], [], [], '--', color='#7894b4', lw=1, label='Quỹ đạo đặt')
        self.trace, = self.ax.plot([], [], [], color=ACCENT, lw=2.4, label='Vị trí từ FK')
        self.ax.legend(loc='lower left', frameon=False, labelcolor=MUTED, fontsize=9)
        self.stats = self.fig.text(.65, .88, '', color=TEXT, fontsize=11, linespacing=1.8,
                                   va='top', family='DejaVu Sans')
        self.chart = self.fig.add_axes((.68, .59, .285, .165), facecolor=PANEL)
        self.chart.set(xlabel='Thời gian mô phỏng (s)', ylabel='Góc (°)')
        self.chart.tick_params(labelsize=8)
        self.chart.grid(color='#2b405b', alpha=.55)
        self.joint_curves = [self.chart.plot([], [], color=c, lw=1.8, label=f'θ{i + 1}')[0]
                             for i, c in enumerate(COLORS)]
        self.cursor = self.chart.axvline(0, color='white', alpha=.55, lw=1)
        self.chart.legend(loc='upper right', ncol=3, fontsize=8, frameon=False,
                          labelcolor=TEXT, handlelength=1)
        self.status = self.fig.text(.65, .13, 'Sẵn sàng · IK → FK', color=ACCENT, fontsize=10)

    def _build_controls(self):
        mode_ax = self.fig.add_axes((.65, .385, .15, .145), facecolor=PANEL)
        path_ax = self.fig.add_axes((.815, .385, .15, .145), facecolor=PANEL)
        self.mode_radio = RadioButtons(mode_ax, (MODE_PATH, MODE_IK, MODE_FK), activecolor=ACCENT)
        self.path_radio = RadioButtons(path_ax, PATH_LABELS,
                                      active=PATH_NAMES.index(self.path_name) if self.path_name in PATH_NAMES else 0,
                                      activecolor=ACCENT)
        for label in (*self.mode_radio.labels, *self.path_radio.labels):
            label.set_color(TEXT)
            label.set_fontsize(10)
        self.mode_radio.on_clicked(self.set_mode)
        self.path_radio.on_clicked(self._select_path)
        self._configure_sliders()
        self.buttons = []
        for x, text, callback in ((.65, 'Tạm dừng', self._toggle_pause),
                                  (.76, 'Chạy lại', self._reset),
                                  (.87, 'Lưu CSV', self._export)):
            ax = self.fig.add_axes((x, .055, .095, .045))
            button = Button(ax, text, color=PANEL, hovercolor='#28415a')
            button.label.set_color(TEXT)
            button.on_clicked(callback)
            self.buttons.append(button)

    def _configure_sliders(self):
        for slider in self.sliders:
            slider.disconnect_events()
            self.fig.delaxes(slider.ax)
        self.sliders = []
        g = self.robot.geometry
        if self.mode == MODE_FK:
            specs = [(f'θ{i + 1} (°)', g.min_angle, g.max_angle, a)
                     for i, a in enumerate(self.pose.angles)]
        elif self.mode == MODE_IK:
            extent = g.base_radius + g.upper_arm
            specs = [('X (mm)', -extent, extent, self.pose.actual[0]),
                     ('Y (mm)', -extent, extent, self.pose.actual[1]),
                     ('Z (mm)', -g.upper_arm - g.lower_arm, -1, self.pose.actual[2])]
        else:
            specs = [('R (mm)', min(10, self.radius), max(120, self.radius), self.radius),
                     ('Z (mm)', min(-400, self.z), max(-180, self.z), self.z),
                     ('Tốc độ', .25, 3, self.speed)]
        for index, (label, low, high, value) in enumerate(specs):
            ax = self.fig.add_axes((.715, .315 - index * .057, .215, .022), facecolor=PANEL)
            slider = Slider(ax, label, low, high, valinit=value, color=ACCENT, valfmt='%1.1f')
            slider.label.set_color(MUTED)
            slider.valtext.set_color(TEXT)
            slider.label.set_fontsize(9)
            slider.on_changed(self._on_slider)
            self.sliders.append(slider)

    def _refresh_path(self):
        points = np.array([p.target for p in self.poses])
        self.path_line.set_data_3d(*points.T)
        angles = np.array([p.angles for p in self.poses])
        times = np.linspace(0, self.duration, len(self.poses))
        for i, line in enumerate(self.joint_curves):
            line.set_data(times, angles[:, i])
        self.chart.set_xlim(0, self.duration)
        self.chart.set_ylim(float(angles.min()) - 8, float(angles.max()) + 8)
        self.closed_path = math.dist(self.poses[0].target, self.poses[-1].target) < 1e-8

    def _paint(self, pose):
        self.pose = pose
        base, elbows, platform = self.robot.joints(pose.angles, pose.actual)
        for i, u in enumerate(self.robot.directions):
            self.upper_lines[i].set_data_3d(*np.array([base[i], elbows[i]]).T)
            tangent = np.array([-u[1], u[0], 0]) * 7
            for side, line in zip((-1, 1), self.lower_lines[i]):
                line.set_data_3d(*np.array([np.array(elbows[i]) + side * tangent,
                                           np.array(platform[i]) + side * tangent]).T)
        self.platform.set_verts([platform])
        self.tool.set_data_3d(*[[v] for v in pose.actual])
        x, y, z = pose.actual
        a, b, c = pose.angles
        self.stats.set_text(f'XYZ  {x:7.2f}  {y:7.2f}  {z:7.2f} mm\n'
                            f'θ    {a:7.2f}  {b:7.2f}  {c:7.2f} °\n'
                            f'Sai số IK/FK   {pose.error:.2e} mm')

    def render_frame(self, index):
        self.index = max(0, min(int(index), len(self.poses) - 1))
        self._paint(self.poses[self.index])
        self.trace.set_data_3d(*np.array([p.actual for p in self.poses[:self.index + 1]]).T)
        t = self.duration * self.index / max(1, len(self.poses) - 1)
        self.cursor.set_xdata([t, t])

    def _message(self, text, error=False):
        self.status.set_text(textwrap.fill(text, width=53))
        self.status.set_color('#ff8993' if error else ACCENT)

    def set_mode(self, label):
        if self.mode_radio.value_selected != label:
            self.mode_radio.set_active((MODE_PATH, MODE_IK, MODE_FK).index(label))
            return
        self.mode = label
        self._invalid_controls = False
        self.last_tick = time.monotonic()
        self._configure_sliders()
        if label == MODE_PATH:
            self._refresh_path()
            self.render_frame(self.index)
        else:
            self.path_line.set_data_3d([], [], [])
            self.trace.set_data_3d([], [], [])
            self._manual_chart()
        self._message(f'{label} · thay đổi bằng thanh trượt')
        self.fig.canvas.draw_idle()

    def _manual_chart(self):
        for angle, line in zip(self.pose.angles, self.joint_curves):
            line.set_data([0, self.duration], [angle, angle])
        self.chart.set_ylim(min(self.pose.angles) - 8, max(self.pose.angles) + 8)
        self.cursor.set_xdata([0, 0])

    def _on_slider(self, _value):
        values = tuple(float(slider.val) for slider in self.sliders)
        try:
            if self.mode == MODE_PATH:
                radius, z, speed = values
                if self.path_name == 'csv':
                    self.speed = speed
                    self._message('CSV: giữ nguyên XYZ; thanh tốc độ điều chỉnh phát lại.')
                    return
                poses = solve_path(self.robot, make_path(self.path_name, radius, z,
                                                        max(5, len(self.poses))))
                self.radius, self.z, self.speed = radius, z, speed
                self.poses = poses
                self._refresh_path()
                self.render_frame(self.index)
            elif self.mode == MODE_IK:
                angles = self.robot.inverse(values)
                self._paint(Pose(values, self.robot.forward(angles), angles))
                self._manual_chart()
            else:
                actual = self.robot.forward(values)
                self._paint(Pose(actual, actual, values))
                self._manual_chart()
            self._invalid_controls = False
            self._message('Hợp lệ · IK → FK' if self.mode != MODE_FK else 'Hợp lệ · FK từ ba góc khớp')
        except KinematicsError as exc:
            self._invalid_controls = True
            self._message(f'Khong hop le: {exc}', error=True)
        self.fig.canvas.draw_idle()

    def _select_path(self, label):
        name = PATH_NAMES[PATH_LABELS.index(label)]
        try:
            poses = solve_path(self.robot, make_path(name, self.radius, self.z,
                                                    max(5, len(self.poses))))
        except KinematicsError as exc:
            if self.path_name in PATH_NAMES:
                self.path_radio.eventson = False
                try:
                    self.path_radio.set_active(PATH_NAMES.index(self.path_name))
                finally:
                    self.path_radio.eventson = True
            self._message(str(exc), error=True)
            return
        self.path_name, self.poses = name, poses
        self.elapsed = 0
        self.index = 0
        self.set_mode(MODE_PATH)

    def _toggle_pause(self, _event=None):
        self.paused = not self.paused
        self.last_tick = time.monotonic()
        self.buttons[0].label.set_text('Tiếp tục' if self.paused else 'Tạm dừng')
        self.fig.canvas.draw_idle()

    def _reset(self, _event=None):
        self.elapsed = 0
        self.index = 0
        self.last_tick = time.monotonic()
        if self.mode == MODE_PATH:
            self._invalid_controls = False
            self._configure_sliders()
            self._message('Chạy lại quỹ đạo hợp lệ gần nhất')
            if self.paused:
                self._toggle_pause()
            self.render_frame(0)
        else:
            for slider in self.sliders:
                slider.reset()
        self.fig.canvas.draw_idle()

    def _export(self, _event=None):
        try:
            path = Path('outputs/motion.csv')
            write_csv(path, self.poses if self.mode == MODE_PATH else [self.pose], self.duration)
            self._message(f'Đã lưu {path}')
        except OSError as exc:
            self._message(f'Không lưu được CSV: {exc}', error=True)
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == ' ':
            self._toggle_pause()
        elif event.key == 'r':
            self._reset()

    def _tick(self, _frame):
        now = time.monotonic()
        delta = now - self.last_tick
        self.last_tick = now
        if self.mode == MODE_PATH and not self.paused and not self._invalid_controls:
            self.elapsed += delta * self.speed
            if self.closed_path:
                self.elapsed %= self.duration
            else:
                self.elapsed = min(self.elapsed, self.duration)
            index = round(self.elapsed / self.duration * (len(self.poses) - 1))
            self.render_frame(index)
            if not self.closed_path and self.elapsed >= self.duration and not self.paused:
                self._toggle_pause()
        return []

    def save_gif(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # GIF stores time in 10 ms ticks. Alternate frame delays so 30 FPS does
        # not silently become 33.3 FPS and shorten the requested cycle by 10%.
        ticks = max(1, round(self.duration * 100))
        count = max(1, min(ticks, round(self.duration * self.fps)))
        delays = [10 * (round((i + 1) * ticks / count) - round(i * ticks / count))
                  for i in range(count)]
        frames = []
        original_dpi, original_index = self.fig.dpi, self.index
        try:
            self.fig.set_dpi(80)
            for i in range(count):
                self.render_frame(round(i / max(1, count - 1) * (len(self.poses) - 1)))
                self.fig.canvas.draw()
                rgb = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()
                frames.append(Image.fromarray(rgb).convert('P', palette=Image.Palette.ADAPTIVE,
                                                          colors=128))
            loop = {'loop': 0} if self.closed_path else {}
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=delays,
                           disposal=2, **loop)
        finally:
            self.fig.set_dpi(original_dpi)
            self.render_frame(original_index)

    def show(self):
        self.last_tick = time.monotonic()
        self.animation = FuncAnimation(self.fig, self._tick, interval=1000 / self.fps,
                                       cache_frame_data=False, blit=False)
        plt.show()
