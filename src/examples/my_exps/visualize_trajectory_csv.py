"""用 LW 模型播放本目录导出的无表头轨迹 CSV（不运行优化器）。

CSV: x,y,z,qx,qy,qz,qw，随后为 hip/thigh/shank/foot/wheel 的右、左关节角。
位置单位 m，角度单位 rad；没有时间列，默认采样率 60 Hz。

在 se3_trajopt 环境中运行：
  python src/examples/my_exps/visualize_trajectory_csv.py
  python src/examples/my_exps/visualize_trajectory_csv.py path/to/trajectory.csv --speed 0.25
  python src/examples/my_exps/visualize_trajectory_csv.py path/to/trajectory.csv --html /tmp/trajectory.html

浏览器 Animation 控件可暂停、播放和拖动时间轴。--html 导出后退出。
这是 CSV 位姿的运动学回放，不是接触动力学仿真；不修正高度或关节角。
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np


SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
JOINT_NAMES = tuple(
    f"{side}_{part}_joint"
    for part in ("hip", "thigh", "shank", "foot", "wheel")
    for side in ("right", "left")
)


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须为有限正数")
    return number


def load_configurations(csv_path, model):
    """按关节名称重排；仅归一化 CSV 舍入造成的微小四元数误差。"""
    rows = np.loadtxt(csv_path, delimiter=",", ndmin=2)
    if rows.shape[0] == 0 or rows.shape[1] != 17:
        raise ValueError(f"需要非空、无表头的 17 列位姿 CSV，实际形状 {rows.shape}")
    if not np.isfinite(rows).all():
        raise ValueError("CSV 含 NaN 或 Inf")
    norms = np.linalg.norm(rows[:, 3:7], axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-3):
        raise ValueError("四元数不是单位四元数；请检查第 4–7 列是否为 qx,qy,qz,qw")
    if model.nq != 17 or model.joints[1].nq != 7:
        raise ValueError("需要浮动基座加 10 个单自由度关节的 LW 模型")
    configurations = np.empty((len(rows), model.nq))
    configurations[:, :7] = rows[:, :7]
    configurations[:, 3:7] /= norms[:, None]
    for column, name in enumerate(JOINT_NAMES, start=7):
        if not model.existJointName(name):
            raise ValueError(f"模型缺少关节 {name}")
        joint = model.joints[model.getJointId(name)]
        if joint.nq != 1:
            raise ValueError(f"关节 {name} 不是单自由度配置")
        configurations[:, joint.idx_q] = rows[:, column]
    return configurations


def create_scene(robot, configurations, fps, speed, loop):
    import meshcat.geometry as geometry
    from meshcat.animation import Animation
    from pinocchio.visualize import MeshcatVisualizer

    vis = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    vis.initViewer(open=False)
    vis.loadViewerModel("LW")
    viewer = vis.viewer

    # 地板上表面严格位于世界系 z=0，不抬高或裁剪 CSV 的机器人位置。
    center = configurations[:, :2].mean(axis=0)
    size = np.maximum(np.ptp(configurations[:, :2], axis=0) + 2.0, 3.0)
    viewer["ground"].set_object(
        geometry.Box([float(size[0]), float(size[1]), 0.02]),
        geometry.MeshLambertMaterial(color=0xC8CDD3, opacity=0.45, transparent=True),
    )
    transform = np.eye(4)
    transform[:3, 3] = [center[0], center[1], -0.01]
    viewer["ground"].set_transform(transform)
    viewer["base_path"].set_object(geometry.Line(
        geometry.PointsGeometry(configurations[:, :3].T),
        geometry.LineBasicMaterial(color=0x246BCE, linewidth=2),
    ))
    # 兼容项目中的旧 Meshcat（没有 set_cam_target 接口）。
    # 平移默认相机整体，保留它原有的观察方向。
    camera_transform = np.eye(4)
    camera_transform[:3, 3] = [center[0], center[1], 0.4]
    viewer["/Cameras/default"].set_transform(camera_transform)

    # 使用 Pinocchio 自身的 display 保留各 mesh 的缩放和局部变换。
    # 录入每个 CSV 帧；无需在 Python 中逐帧 sleep，浏览器负责播放。
    animation = Animation(default_framerate=fps * speed)
    try:
        for frame, q in enumerate(configurations):
            with animation.at_frame(viewer, frame) as frame_viewer:
                vis.viewer = frame_viewer
                vis.display(q)
    finally:
        vis.viewer = viewer
    vis.display(configurations[0])
    viewer.set_animation(animation, play=True, repetitions=1_000_000 if loop else 1)
    return vis


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", type=Path,
                        default=Path(__file__).with_name("leg_to_wheel_transform_60hz.csv"))
    parser.add_argument("--fps", type=positive_float, default=60.0, help="CSV 原始采样率（默认 60）")
    parser.add_argument("--speed", type=positive_float, default=1.0, help="播放倍速，例如 0.25")
    parser.add_argument("--loop", action="store_true", help="重复播放；每次循环跳回首帧")
    parser.add_argument("--no-browser", action="store_true", help="仅打印网址，不自动打开浏览器")
    parser.add_argument("--html", type=Path, help="保存含模型与动画的 HTML，然后退出")
    parser.add_argument("--check-only", action="store_true", help="仅检查 CSV、模型映射和正运动学")
    args = parser.parse_args()
    from robots.LW_Leg.LWLegWrapper import LW_Leg

    robot = LW_Leg()
    try:
        configurations = load_configurations(args.csv, robot.model)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"CSV: {args.csv.resolve()}")
    print(f"{len(configurations)} 帧，{args.fps:g} Hz，首末帧间隔 "
          f"{(len(configurations) - 1) / args.fps:.3f} s，播放速度 {args.speed:g} 倍")
    print("世界系 Z 向上；四元数 xyzw；关节按名称重排。仅回放位姿，不计算力矩或冲击力。")
    if args.check_only:
        for q in configurations:
            robot.fk_all(q)
            if not all(np.isfinite(placement.homogeneous).all() for placement in robot.data.oMf):
                raise ValueError("正运动学产生非有限值")
        print("CSV 校验、关节映射与全部帧正运动学通过。")
        return

    vis = create_scene(robot, configurations, args.fps, args.speed, args.loop)
    try:
        if args.html:
            args.html.parent.mkdir(parents=True, exist_ok=True)
            args.html.write_text(vis.viewer.static_html(), encoding="utf-8")
            print(f"已导出 HTML: {args.html.resolve()}，用浏览器打开即可播放。")
            return
        print(f"浏览器地址: {vis.viewer.url()}", flush=True)
        print("展开 Animation 控件可暂停及拖动时间轴；终端 Ctrl+C 退出。", flush=True)
        if not args.no_browser:
            vis.viewer.open()
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n已停止。")
    # Meshcat 在 atexit 中回收本脚本启动的服务。当前版本的
    # Visualizer.close() 调用了不存在的 ViewerWindow.close()，故不调用它。


if __name__ == "__main__":
    main()
