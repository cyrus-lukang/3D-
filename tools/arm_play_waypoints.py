#!/usr/bin/env python3
"""按录制的路点扫过，供 nvblox 建图。

默认用 rm_movej_p（关节空间到位）——示教点之间笛卡尔直线常过奇异/超限，
导致「前几个点走了、某一点卡住不动」。需要直线时加 --linear。

默认 dry-run。真正运动：--execute 且输入 YES。Ctrl+C → 急停。

  export CALIBRATION_ROOT=/home/ubuntu/workspace/lk_ceshi/Calibration
  python3 tools/arm_play_waypoints.py --execute --speed 15
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

Pose = List[float]


def _add_calibration_path() -> Optional[Path]:
    import os
    roots = []
    env = os.environ.get('CALIBRATION_ROOT')
    if env:
        roots.append(Path(env))
    here = Path(__file__).resolve()
    roots += [
        here.parents[2] / 'Calibration',
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


def _fmt(pose: Pose) -> str:
    return ('xyz=[' + ', '.join(f'{v: .4f}' for v in pose[:3]) + ']  '
            'rpy=[' + ', '.join(f'{v: .4f}' for v in pose[3:]) + ']')


def load_poses(path: Path) -> List[Pose]:
    data = json.loads(path.read_text(encoding='utf-8'))
    poses = []
    for item in data.get('poses', []):
        xyzrpy = item.get('xyzrpy')
        if xyzrpy is None or len(xyzrpy) != 6:
            raise SystemExit(f'坏点: {item}')
        poses.append([float(v) for v in xyzrpy])
    if not poses:
        raise SystemExit(f'{path} 没有 poses')
    return poses


def clear_err(arm) -> None:
    robot = getattr(arm, '_robot', None)
    if robot is None:
        return
    try:
        ret = robot.rm_clear_system_err()
        print(f'  rm_clear_system_err → {ret}', flush=True)
    except Exception as exc:  # pylint: disable=broad-except
        print(f'  clear_err 异常: {exc}', flush=True)


def move_to(arm, pose: Pose, speed: float, linear: bool) -> bool:
    """到位。linear=True 用 rm_movel，失败则清错后改 movej_p。"""
    robot = arm._robot  # noqa: SLF001
    if robot is None:
        return False

    if linear:
        try:
            ret = robot.rm_movel(
                pose=pose, v=int(round(speed)), r=0, connect=0, block=1)
            if ret == 0:
                return True
            print(f'  rm_movel 失败 ret={ret}，清错后改用 movej_p ...', flush=True)
            clear_err(arm)
            time.sleep(0.2)
        except Exception as exc:  # pylint: disable=broad-except
            print(f'  rm_movel 异常: {exc}，清错后改用 movej_p ...', flush=True)
            clear_err(arm)
            time.sleep(0.2)

    ok = arm.move_to_pose(pose, speed=speed)
    if ok:
        return True
    print('  movej_p 失败，清错再试一次 ...', flush=True)
    clear_err(arm)
    time.sleep(0.3)
    return arm.move_to_pose(pose, speed=speed)


def _sample_joints_hz(arm_ip: str, arm_port: int, hz: float, stop: threading.Event,
                      rows: list, lock: threading.Lock) -> None:
    """运动过程中另开一条只读连接，按 hz 持续记关节。碎片不能只靠到位那几帧。"""
    if _add_calibration_path() is None:
        print('持续采样：找不到 Calibration', flush=True)
        return
    from modules.robot.realman.legacy_arm_controller import ArmController  # type: ignore
    reader = ArmController(arm_ip, arm_port)
    if not reader.connect():
        print('持续采样：连接失败，只保留到位关节', flush=True)
        return
    period = 1.0 / max(hz, 0.1)
    t0 = time.time()
    n = 0
    try:
        while not stop.is_set():
            t_loop = time.time()
            ok, jdeg = reader.get_current_joints()
            if ok and jdeg is not None and len(jdeg) == 7:
                rec = {
                    't': round(t_loop - t0, 3),
                    'joints_deg': [round(float(v), 3) for v in jdeg],
                }
                with lock:
                    rows.append(rec)
                n += 1
            wait = period - (time.time() - t_loop)
            if stop.wait(max(0.0, wait)):
                break
    finally:
        try:
            reader.disconnect()
        except Exception:
            pass
        print(f'持续采样结束 {n} 帧 @ {hz:g} Hz', flush=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description='Play recorded waypoints.')
    p.add_argument('--arm-ip', default='192.168.1.18')
    p.add_argument('--arm-port', type=int, default=8080)
    p.add_argument(
        '--waypoints',
        default=str(root / 'gpu_logs' / 'scan_waypoints_latest.json'),
        help='JSON from arm_record_waypoints.py (your taught poses)',
    )
    p.add_argument('--speed', type=float, default=15.0,
                   help='速度百分比 1-100（默认 15）')
    p.add_argument('--dwell', type=float, default=0.3,
                   help='每点到位后停留秒数')
    p.add_argument('--loop', type=int, default=1, help='整条路径重复次数')
    p.add_argument('--linear', action='store_true',
                   help='用 rm_movel 笛卡尔直线（易在跨奇异段失败；默认关）')
    p.add_argument('--log-joints', type=str, default='',
                   help='运动全程按 --log-hz 记录 J1～J7(度)，供剔臂并集')
    p.add_argument('--log-hz', type=float, default=10.0,
                   help='记录过程持续采样频率（默认 10 Hz）')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--yes', action='store_true')
    args = p.parse_args()

    if not (0 < args.speed <= 100):
        raise SystemExit('--speed must be in (0, 100]')

    wp_path = Path(args.waypoints)
    if not wp_path.is_file():
        raise SystemExit(f'找不到路点文件: {wp_path}\n请先跑 arm_record_waypoints.py')

    poses = load_poses(wp_path)
    mode = 'rm_movel(+movej_p fallback)' if args.linear else 'rm_movej_p'
    print(f'路点文件: {wp_path}  ({len(poses)} 点)  mode={mode}')
    for i, pose in enumerate(poses):
        print(f'  p{i + 1}: {_fmt(pose)}')
        if i > 0:
            a, b = poses[i - 1], pose
            dist = sum((b[j] - a[j]) ** 2 for j in range(3)) ** 0.5
            print(f'       ← 距上点 {dist * 1000:.0f} mm')

    if not args.execute:
        print('\n[DRY-RUN] 未运动。确认后加 --execute。')
        return 0

    if _add_calibration_path() is None:
        raise SystemExit('找不到 Calibration，请 export CALIBRATION_ROOT=...')

    from modules.robot.realman.legacy_arm_controller import ArmController  # type: ignore

    if not args.yes:
        ans = input(f'将用 {mode} @ {args.speed}% 扫 {len(poses)} 点×{args.loop} 圈。'
                    '输入 YES 继续: ').strip()
        if ans != 'YES':
            print('已取消。')
            return 1

    arm = ArmController(args.arm_ip, args.arm_port)
    if not arm.connect():
        raise SystemExit(f'连接失败 {args.arm_ip}:{args.arm_port}')

    stopped = {'v': False}
    sample_stop = threading.Event()
    sample_rows: list = []
    sample_lock = threading.Lock()
    sampler = None
    if args.log_joints:
        sampler = threading.Thread(
            target=_sample_joints_hz,
            args=(args.arm_ip, args.arm_port, args.log_hz, sample_stop,
                  sample_rows, sample_lock),
            daemon=True,
        )
        sampler.start()
        print(f'关节持续采样 {args.log_hz:g} Hz → {args.log_joints}', flush=True)

    def _flush_joint_log() -> None:
        if not args.log_joints:
            return
        sample_stop.set()
        if sampler is not None:
            sampler.join(timeout=2.0)
        with sample_lock:
            rows = list(sample_rows)
        jp = Path(args.log_joints)
        jp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'unit': 'deg',
            'hz': args.log_hz,
            'count': len(rows),
            'poses': rows,
        }
        jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                      encoding='utf-8')
        print(f'已写关节日志 {len(rows)} 帧 @ {args.log_hz:g} Hz → {jp}', flush=True)

    def _on_sigint(sig, frame):  # noqa: ANN001
        print('\n>>> SIGINT：急停 ...', flush=True)
        sample_stop.set()
        try:
            arm.emergency_stop()
        except Exception as exc:  # pylint: disable=broad-except
            print(f'急停失败: {exc}', flush=True)
        stopped['v'] = True

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        clear_err(arm)
        t0 = time.time()
        for lap in range(args.loop):
            print(f'--- 第 {lap + 1}/{args.loop} 圈 ---')
            for i, pose in enumerate(poses):
                if stopped['v']:
                    return 2
                print(f'→ p{i + 1}/{len(poses)} {_fmt(pose)}', flush=True)
                if not move_to(arm, pose, args.speed, linear=args.linear):
                    ok, cur = arm.get_current_pose()
                    if ok:
                        print(f'  当前位姿 {_fmt([float(v) for v in cur])}')
                    print('运动失败。请示教器确认该点可达，或减小跨度后重录。')
                    arm.emergency_stop()
                    return 3
                print(f'  ✓ p{i + 1} 到位', flush=True)
                if args.log_joints:
                    jok, jdeg = arm.get_current_joints()
                    if jok and jdeg is not None and len(jdeg) == 7:
                        with sample_lock:
                            sample_rows.append({
                                't': round(time.time() - t0, 3),
                                'name': f'p{i + 1}_lap{lap + 1}',
                                'joints_deg': [round(float(v), 3) for v in jdeg],
                            })
                time.sleep(max(0.0, args.dwell))
        print(f'完成，用时 {time.time() - t0:.1f}s')
        return 0
    finally:
        if args.log_joints:
            _flush_joint_log()
        arm.disconnect()


if __name__ == '__main__':
    raise SystemExit(main())
