#!/usr/bin/env python3
"""只转末端关节（默认 J7）指定角度，其它关节保持不动。

默认 dry-run：只打印目标。真正运动必须：
  1) 加 --execute
  2) 终端输入 YES（或加 --yes）

示例（先看目标）：
  python3 tools/arm_rotate_ee_joint.py --arm-ip 192.168.1.18

转 180° 并执行：
  python3 tools/arm_rotate_ee_joint.py --arm-ip 192.168.1.18 \\
    --degrees 180 --speed 8 --execute

转完再转回原位：
  python3 tools/arm_rotate_ee_joint.py --degrees 180 --return --execute
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _add_calibration_path() -> Optional[Path]:
    import os

    roots = []
    env = os.environ.get("CALIBRATION_ROOT")
    if env:
        roots.append(Path(env))
    here = Path(__file__).resolve()
    roots += [
        here.parents[2] / "Calibration",
        Path("/workspaces/Calibration"),
        Path("/home/ubuntu/workspace/lk_ceshi/Calibration"),
    ]
    for root in roots:
        if (
            root
            / "calibration_runtime"
            / "modules"
            / "robot"
            / "realman"
            / "legacy_arm_controller.py"
        ).is_file():
            sys.path.insert(0, str(root))
            sys.path.insert(0, str(root / "calibration_runtime"))
            return root
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rotate only the EE joint (J7 by default); hold other joints."
    )
    p.add_argument("--arm-ip", default="192.168.1.18")
    p.add_argument("--arm-port", type=int, default=8080)
    p.add_argument(
        "--joint",
        type=int,
        default=7,
        help="1-based joint index to rotate (default 7 = EE).",
    )
    p.add_argument(
        "--degrees",
        type=float,
        default=180.0,
        help="Relative rotation in degrees (default +180).",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=8.0,
        help="rm_movej speed percent (1-100). Default 8 (slow).",
    )
    p.add_argument(
        "--dwell",
        type=float,
        default=0.5,
        help="Seconds to wait after each move.",
    )
    p.add_argument(
        "--margin-deg",
        type=float,
        default=2.0,
        help="Keep this many degrees away from soft limits.",
    )
    p.add_argument(
        "--return",
        dest="do_return",
        action="store_true",
        help="After rotating, move back to the start joint angles.",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually move. Without this, only plan/print.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive YES prompt (still needs --execute).",
    )
    return p.parse_args()


def _pick_target(
    current: float,
    delta: float,
    lower: float,
    upper: float,
    margin: float,
) -> Tuple[float, str]:
    """Choose target = current+delta, or flip sign if that hits the soft limit."""
    soft_lo = lower + margin
    soft_hi = upper - margin
    if soft_lo >= soft_hi:
        raise SystemExit(f"限位过窄: [{lower}, {upper}] margin={margin}")

    cand = current + delta
    if soft_lo <= cand <= soft_hi:
        return cand, f"current{delta:+.1f}°"

    alt = current - delta
    if soft_lo <= alt <= soft_hi:
        return alt, f"限位翻转 → current{-delta:+.1f}°"

    # Clamp toward the nearer feasible side of the requested delta.
    clamped = min(max(cand, soft_lo), soft_hi)
    note = (
        f"超出软限位，钳位到 {clamped:.2f}° "
        f"(请求 {cand:.2f}°，软区 [{soft_lo:.1f},{soft_hi:.1f}])"
    )
    if abs(clamped - current) < 1e-3:
        raise SystemExit(
            f"J 目标无法移动：当前={current:.2f}° 请求Δ={delta:.1f}° "
            f"软限位=[{soft_lo:.1f},{soft_hi:.1f}]"
        )
    return clamped, note


def main() -> int:
    args = parse_args()
    if not (0 < args.speed <= 100):
        raise SystemExit("--speed must be in (0, 100]")
    if not (1 <= args.joint <= 7):
        raise SystemExit("--joint must be in 1..7")

    root = _add_calibration_path()
    if root is None:
        raise SystemExit(
            "找不到 Calibration（需要 legacy_arm_controller）。"
            "请 export CALIBRATION_ROOT=.../Calibration"
        )

    from modules.robot.realman.legacy_arm_controller import ArmController  # type: ignore

    arm = ArmController(args.arm_ip, args.arm_port)
    if not arm.connect():
        raise SystemExit(f"连接机械臂失败 {args.arm_ip}:{args.arm_port}")

    stopped = {"v": False}

    def _on_sigint(sig, frame):  # noqa: ANN001
        print("\n>>> SIGINT：尝试急停 rm_set_arm_stop ...", flush=True)
        try:
            arm.emergency_stop()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"急停失败: {exc}", flush=True)
        stopped["v"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        ok, joints = arm.get_current_joints()
        if not ok:
            raise SystemExit(f"读当前关节失败: {joints}")
        start: List[float] = [float(v) for v in joints]
        if len(start) != 7:
            raise SystemExit(f"期望 7 轴，得到 {len(start)}: {start}")

        limits = arm.get_joint_limits()
        ji = args.joint - 1
        lower, upper = limits[ji]
        target_j, how = _pick_target(
            start[ji], args.degrees, lower, upper, args.margin_deg
        )

        goal = list(start)
        goal[ji] = target_j

        print(f"当前关节 (deg): {[round(v, 2) for v in start]}")
        print(
            f"只动 J{args.joint}: {start[ji]:.2f}° → {target_j:.2f}°  "
            f"({how})  限位=[{lower:.1f},{upper:.1f}]"
        )
        print(f"目标关节 (deg): {[round(v, 2) for v in goal]}")
        print(f"其它关节保持不变。speed={args.speed}%  return={args.do_return}")

        # Sanity: only one joint should change
        deltas = [abs(g - s) for g, s in zip(goal, start)]
        movers = [i + 1 for i, d in enumerate(deltas) if d > 1e-6]
        print(f"将变化的关节: {movers}  Δ={[round(d, 2) for d in deltas]}")

        path: List[Tuple[str, Sequence[float]]] = [("rotate", goal)]
        if args.do_return:
            path.append(("return", start))

        if not args.execute:
            print("\n[DRY-RUN] 未运动。确认后加 --execute（并准备好急停）。")
            return 0

        if not args.yes:
            ans = input("将开始运动。输入 YES 继续，其它键取消: ").strip()
            if ans != "YES":
                print("已取消。")
                return 1

        print("开始运动。Ctrl+C = 急停。")
        t0 = time.time()
        for name, q in path:
            if stopped["v"]:
                return 2
            print(
                f"→ {name}  J{args.joint}={q[ji]:.2f}°  "
                f"q={[round(v, 2) for v in q]}",
                flush=True,
            )
            if not arm.move_to_joints(q, args.speed, label=f"EE旋转-{name}"):
                print("运动失败，急停。")
                arm.emergency_stop()
                return 4
            time.sleep(max(0.0, args.dwell))

        ok2, after = arm.get_current_joints()
        if ok2:
            print(f"到位关节 (deg): {[round(v, 2) for v in after]}")
        print(f"完成，用时 {time.time() - t0:.1f}s")
        return 0
    finally:
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
