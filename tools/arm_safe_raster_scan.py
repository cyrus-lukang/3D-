#!/usr/bin/env python3
"""RealMan 安全盒内弓字扫视（给 nvblox 建图用）。

默认只打印路径（dry-run）。真正运动必须同时：
  1) 加 --execute
  2) 终端输入 YES

安全：
  - 所有目标必须落在 Base 系安全盒 [xmin,xmax]×[ymin,ymax]×[zmin,zmax]
  - 姿态默认锁定为启动时的 EE 姿态（只平移扫）
  - 速度默认很低（movej_p 百分比，建议 ≤15）
  - Ctrl+C / 失败会尝试 rm_set_arm_stop

依赖：现场已安装 RealMan SDK（Robotic_Arm），并挂载/可见 Calibration。

示例（主机，先看路径）：
  export CALIBRATION_ROOT=/home/ubuntu/workspace/lk_ceshi/Calibration
  python3 tools/arm_safe_raster_scan.py --arm-ip 192.168.1.18

相对当前位姿划盒并执行：
  python3 tools/arm_safe_raster_scan.py --arm-ip 192.168.1.18 \\
    --box-from-current --dx 0.08 --dy 0.10 --step 0.04 --speed 10 --execute
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

Pose = List[float]  # [x,y,z,rx,ry,rz] meters + radians


def _add_calibration_path() -> Optional[Path]:
    import os
    roots = []
    env = os.environ.get('CALIBRATION_ROOT')
    if env:
        roots.append(Path(env))
    here = Path(__file__).resolve()
    roots += [
        here.parents[2] / 'Calibration',  # lk_ceshi/Calibration
        Path('/workspaces/Calibration'),
        Path('/home/ubuntu/workspace/lk_ceshi/Calibration'),
    ]
    for root in roots:
        if (root / 'calibration_runtime' / 'modules' / 'robot' / 'realman' /
                'legacy_arm_controller.py').is_file():
            sys.path.insert(0, str(root))
            sys.path.insert(0, str(root / 'calibration_runtime'))
            return root
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Safe-box lawnmower scan for nvblox (RealMan).')
    p.add_argument('--arm-ip', default='192.168.1.18')
    p.add_argument('--arm-port', type=int, default=8080)
    p.add_argument('--speed', type=float, default=10.0,
                   help='rm_movej_p speed percent (1-100). Default 10 (slow).')
    p.add_argument('--dwell', type=float, default=0.4,
                   help='Seconds to wait at each waypoint after arrival.')
    p.add_argument('--step', type=float, default=0.04,
                   help='Raster step in meters (default 4cm).')
    p.add_argument('--rows', type=int, default=0,
                   help='Override row count; 0 = fill box by --step.')
    p.add_argument('--cols', type=int, default=0,
                   help='Override col count; 0 = fill box by --step.')

    # Absolute box (Base frame, meters). Ignored if --box-from-current.
    p.add_argument('--xmin', type=float, default=None)
    p.add_argument('--xmax', type=float, default=None)
    p.add_argument('--ymin', type=float, default=None)
    p.add_argument('--ymax', type=float, default=None)
    p.add_argument('--zmin', type=float, default=None)
    p.add_argument('--zmax', type=float, default=None)

    p.add_argument('--box-from-current', action='store_true',
                   help='Build box around current EE: ±dx, ±dy, z in [z-dz,z+dz].')
    p.add_argument('--dx', type=float, default=0.06, help='Half-width X (m).')
    p.add_argument('--dy', type=float, default=0.08, help='Half-width Y (m).')
    p.add_argument('--dz', type=float, default=0.0,
                   help='Half-height Z (m). 0 = keep Z fixed.')

    p.add_argument('--pitch-offsets-deg', type=str, default='0',
                   help='Comma list of extra ry offsets in deg, e.g. 0,-10,10')
    p.add_argument('--execute', action='store_true',
                   help='Actually move. Without this, only plan/print.')
    p.add_argument('--yes', action='store_true',
                   help='Skip interactive YES prompt (still needs --execute).')
    return p.parse_args()


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [0.5 * (a + b)]
    return list(np.linspace(a, b, n))


def build_box(args: argparse.Namespace, cur: Pose) -> Tuple[float, float, float, float, float, float]:
    if args.box_from_current:
        x, y, z = cur[0], cur[1], cur[2]
        return (x - args.dx, x + args.dx, y - args.dy, y + args.dy,
                z - args.dz, z + args.dz)
    needed = [args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax]
    if any(v is None for v in needed):
        raise SystemExit(
            '请给绝对安全盒 --xmin/xmax/ymin/ymax/zmin/zmax，'
            '或使用 --box-from-current')
    return (float(args.xmin), float(args.xmax), float(args.ymin), float(args.ymax),
            float(args.zmin), float(args.zmax))


def in_box(x: float, y: float, z: float, box: Sequence[float], eps: float = 1e-6) -> bool:
    xmin, xmax, ymin, ymax, zmin, zmax = box
    return (xmin - eps <= x <= xmax + eps and ymin - eps <= y <= ymax + eps
            and zmin - eps <= z <= zmax + eps)


def plan_raster(box: Sequence[float], step: float, rows: int, cols: int,
                rpy: Sequence[float],
                pitch_offsets_rad: Sequence[float]) -> List[Pose]:
    xmin, xmax, ymin, ymax, zmin, zmax = box
    if cols <= 0:
        cols = max(1, int(round((xmax - xmin) / max(step, 1e-6))) + 1)
    if rows <= 0:
        rows = max(1, int(round((ymax - ymin) / max(step, 1e-6))) + 1)
    xs = _linspace(xmin, xmax, cols)
    ys = _linspace(ymin, ymax, rows)
    # Single Z mid-plane (or fixed if zmin≈zmax)
    z = 0.5 * (zmin + zmax)
    waypoints: List[Pose] = []
    for pi, dpitch in enumerate(pitch_offsets_rad):
        rx, ry, rz = float(rpy[0]), float(rpy[1]) + float(dpitch), float(rpy[2])
        for r, y in enumerate(ys):
            row_xs = xs if (r % 2 == 0) else list(reversed(xs))
            for x in row_xs:
                if not in_box(x, y, z, box):
                    raise RuntimeError(f'planner bug: ({x},{y},{z}) outside box')
                waypoints.append([x, y, z, rx, ry, rz])
        if pi + 1 < len(pitch_offsets_rad):
            # return to start of next pitch layer from last point (already there)
            pass
    return waypoints


def main() -> int:
    args = parse_args()
    if not (0 < args.speed <= 100):
        raise SystemExit('--speed must be in (0, 100]')

    root = _add_calibration_path()
    if root is None:
        raise SystemExit(
            '找不到 Calibration（需要 legacy_arm_controller）。'
            '请 export CALIBRATION_ROOT=.../Calibration')

    from modules.robot.realman.legacy_arm_controller import ArmController  # type: ignore

    pitch_deg = [float(x) for x in args.pitch_offsets_deg.split(',') if x.strip() != '']
    pitch_rad = [np.deg2rad(d) for d in pitch_deg]

    arm = ArmController(args.arm_ip, args.arm_port)
    if not arm.connect():
        raise SystemExit(f'连接机械臂失败 {args.arm_ip}:{args.arm_port}')

    stopped = {'v': False}

    def _on_sigint(sig, frame):  # noqa: ANN001
        print('\n>>> SIGINT：尝试急停 rm_set_arm_stop ...', flush=True)
        try:
            arm.emergency_stop()
        except Exception as exc:  # pylint: disable=broad-except
            print(f'急停失败: {exc}', flush=True)
        stopped['v'] = True

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        ok, cur = arm.get_current_pose()
        if not ok:
            raise SystemExit(f'读当前位姿失败: {cur}')
        cur = [float(v) for v in cur]
        print(f'当前 EE xyzrpy (m,rad) = {[round(v, 4) for v in cur]}')

        box = build_box(args, cur)
        print('安全盒 Base (m): '
              f'x[{box[0]:.3f},{box[1]:.3f}] '
              f'y[{box[2]:.3f},{box[3]:.3f}] '
              f'z[{box[4]:.3f},{box[5]:.3f}]')

        if not in_box(cur[0], cur[1], cur[2], box):
            raise SystemExit(
                '当前 EE 不在安全盒内。请先手动挪进盒子，或加大 --dx/--dy/--dz / 改绝对盒。')

        wps = plan_raster(box, args.step, args.rows, args.cols, cur[3:6], pitch_rad)
        print(f'规划路点数: {len(wps)}  step={args.step}m  speed={args.speed}%  '
              f'dwell={args.dwell}s  pitch_deg={pitch_deg}')
        for i, w in enumerate(wps[:8]):
            print(f'  [{i:03d}] xyz={[round(v, 4) for v in w[:3]]} '
                  f'rpy={[round(v, 4) for v in w[3:]]}')
        if len(wps) > 8:
            print(f'  ... ({len(wps) - 8} more)')

        if not args.execute:
            print('\n[DRY-RUN] 未运动。确认路径后加 --execute（并准备好急停）。')
            return 0

        if not args.yes:
            ans = input('将开始运动。输入 YES 继续，其它键取消: ').strip()
            if ans != 'YES':
                print('已取消。')
                return 1

        print('开始扫视。Ctrl+C = 急停。')
        t0 = time.time()
        for i, pose in enumerate(wps):
            if stopped['v']:
                return 2
            if not in_box(pose[0], pose[1], pose[2], box):
                print(f'拒绝越界点 {pose[:3]}，急停。')
                arm.emergency_stop()
                return 3
            print(f'→ [{i+1}/{len(wps)}] {[round(v, 4) for v in pose[:3]]}', flush=True)
            if not arm.move_to_pose(pose, speed=args.speed):
                print('运动失败，急停。')
                arm.emergency_stop()
                return 4
            time.sleep(max(0.0, args.dwell))
        print(f'完成 {len(wps)} 点，用时 {time.time() - t0:.1f}s')
        return 0
    finally:
        arm.disconnect()


if __name__ == '__main__':
    raise SystemExit(main())
