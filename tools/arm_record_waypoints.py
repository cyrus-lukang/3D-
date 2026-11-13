#!/usr/bin/env python3
"""手动示教：记录 N 个 EE 位姿（Base，米+弧度）供建图扫视回放。

用法（主机）：
  export CALIBRATION_ROOT=/home/ubuntu/workspace/lk_ceshi/Calibration
  python3 tools/arm_record_waypoints.py --count 5

操作：
  用示教器/拖动把臂挪到想要的位置 → 回车记录当前位姿。
  录满后写入 JSON；也可随时输入 q 提前结束并保存已录点。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
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


def main() -> int:
    p = argparse.ArgumentParser(description='Record EE waypoints for nvblox scan.')
    p.add_argument('--arm-ip', default='192.168.1.18')
    p.add_argument('--arm-port', type=int, default=8080)
    p.add_argument('--count', type=int, default=5, help='How many poses to record.')
    p.add_argument(
        '--out',
        default='',
        help='Output JSON path. Default: gpu_logs/scan_waypoints_<timestamp>.json',
    )
    args = p.parse_args()
    if args.count < 1:
        raise SystemExit('--count must be >= 1')

    if _add_calibration_path() is None:
        raise SystemExit('找不到 Calibration，请 export CALIBRATION_ROOT=...')

    from modules.robot.realman.legacy_arm_controller import ArmController  # type: ignore

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else (
        root / 'gpu_logs' /
        f'scan_waypoints_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    out.parent.mkdir(parents=True, exist_ok=True)

    arm = ArmController(args.arm_ip, args.arm_port)
    if not arm.connect():
        raise SystemExit(f'连接失败 {args.arm_ip}:{args.arm_port}')

    poses: List[Pose] = []
    try:
        print(f'将录 {args.count} 个点。示教器挪好后按回车记录；输入 q 结束并保存。')
        print(f'输出: {out}')
        while len(poses) < args.count:
            ok, cur = arm.get_current_pose()
            if not ok:
                print(f'读位姿失败: {cur}')
                return 1
            cur = [float(v) for v in cur]
            n = len(poses) + 1
            print(f'\n[{n}/{args.count}] 当前 {_fmt(cur)}')
            cmd = input('回车=记录此点 / r=刷新显示 / q=结束保存 > ').strip().lower()
            if cmd == 'q':
                break
            if cmd == 'r':
                continue
            if cmd not in ('', 'y', 'yes'):
                print('未识别，已忽略。')
                continue
            poses.append(cur)
            print(f'  ✓ 已记录点{len(poses)}: {_fmt(cur)}')
            time.sleep(0.05)

        if not poses:
            print('未记录任何点，不写文件。')
            return 1

        payload = {
            'unit': 'm_rad',
            'frame': 'base_ee',
            'arm_ip': args.arm_ip,
            'recorded_at': datetime.now().isoformat(timespec='seconds'),
            'count': len(poses),
            'poses': [
                {
                    'name': f'p{i + 1}',
                    'xyzrpy': [round(v, 6) for v in pose],
                }
                for i, pose in enumerate(poses)
            ],
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                       encoding='utf-8')
        # 也写一份固定名，方便回放默认加载
        latest = root / 'gpu_logs' / 'scan_waypoints_latest.json'
        latest.write_text(out.read_text(encoding='utf-8'), encoding='utf-8')

        print(f'\n已保存 {len(poses)} 点 → {out}')
        print(f'同时更新 → {latest}')
        for i, pose in enumerate(poses):
            print(f'  p{i + 1}: {_fmt(pose)}')
        return 0
    finally:
        arm.disconnect()


if __name__ == '__main__':
    raise SystemExit(main())
