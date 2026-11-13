#!/usr/bin/env python3
"""实验 G：用 RM75 URDF 各连杆 AABB 膨胀后，剔除落在膨胀盒内的点云/网格顶点。

流程：
  1) 读 URDF（默认 RM75-B）+ 各 link collision STL
  2) 用当前关节角（或 joints JSON）做 FK，把每个 link 的 AABB 在连杆系内膨胀 inflate_m
  3) 对 PLY 顶点：若落在任一姿态、任一连杆的膨胀长方体内 → 删除

示例：
  # 读当前臂关节，过滤 live_mesh.ply
  python3 tools/arm_self_filter_ply.py \\
    --ply live_mesh.ply --out live_mesh_self_filtered.ply --from-arm

  # 多组关节（扫视过程中录的）取并集剔除
  python3 tools/arm_self_filter_ply.py \\
    --ply live_mesh.ply --joints-json gpu_logs/scan_joints.json --inflate-m 0.05
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = (
    Path('/home/ubuntu/workspace/lk_ceshi/robot_models/rm_models_RM75/'
         'urdf/RM75-B/urdf/RM75-B.urdf')
)
DEFAULT_PLY = ROOT / 'live_mesh.ply'


@dataclass
class LinkBox:
    """连杆坐标系下的轴对齐长方体（已膨胀）。"""
    name: str
    aabb_min: np.ndarray  # (3,)
    aabb_max: np.ndarray  # (3,)


@dataclass
class WorldPoseBox:
    name: str
    T_world_link: np.ndarray  # 4x4
    aabb_min: np.ndarray
    aabb_max: np.ndarray


def _add_calibration_path() -> Optional[Path]:
    import os
    roots = []
    env = os.environ.get('CALIBRATION_ROOT')
    if env:
        roots.append(Path(env))
    roots += [
        ROOT.parents[0] / 'Calibration',
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


def prepare_urdf_abs(urdf_path: Path, out_path: Path) -> Path:
    """把 package://.../ 换成绝对路径，供 pinocchio / trimesh 用。"""
    text = urdf_path.read_text(encoding='utf-8')
    # RM75-B 风格：package://RM75-B/ → 包根（含 meshes/）
    pkg_root = urdf_path.resolve().parents[1]
    for pkg in ('RM75-B', 'RM75-B-V', 'RM75-6F', 'RM75-6FB', 'RM75-6FB-V'):
        text = text.replace(f'package://{pkg}/', str(pkg_root) + '/')
    # ROS2 rm_description：package://rm_description/ → description 包根
    desc_root = urdf_path.resolve().parents[1]
    text = text.replace('package://rm_description/', str(desc_root) + '/')
    out_path.write_text(text, encoding='utf-8')
    return out_path


def load_link_boxes_local(urdf_abs: Path, inflate_m: float) -> List[LinkBox]:
    """从 URDF collision mesh 算每个 link 局部 AABB，再膨胀 inflate_m（每侧）。"""
    import xml.etree.ElementTree as ET
    import trimesh

    root = ET.parse(urdf_abs).getroot()
    boxes: List[LinkBox] = []
    for link in root.findall('link'):
        name = link.get('name') or ''
        mesh_path = None
        origin_xyz = np.zeros(3)
        origin_rpy = np.zeros(3)
        col = link.find('collision')
        if col is not None:
            org = col.find('origin')
            if org is not None:
                if org.get('xyz'):
                    origin_xyz = np.array([float(x) for x in org.get('xyz').split()])
                if org.get('rpy'):
                    origin_rpy = np.array([float(x) for x in org.get('rpy').split()])
            mesh = col.find('geometry/mesh')
            if mesh is not None and mesh.get('filename'):
                mesh_path = Path(mesh.get('filename'))
        if mesh_path is None:
            vis = link.find('visual/geometry/mesh')
            if vis is not None and vis.get('filename'):
                mesh_path = Path(vis.get('filename'))
        if mesh_path is None or not mesh_path.is_file():
            continue
        tm = trimesh.load(str(mesh_path), force='mesh')
        verts = np.asarray(tm.vertices, dtype=np.float64)
        if verts.size == 0:
            continue
        # collision origin: R_rpy * v + t
        R = _rpy_to_R(origin_rpy)
        verts_l = (R @ verts.T).T + origin_xyz
        mn = verts_l.min(axis=0) - inflate_m
        mx = verts_l.max(axis=0) + inflate_m
        boxes.append(LinkBox(name=name, aabb_min=mn, aabb_max=mx))
    return boxes


def _rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    rx, ry, rz = rpy
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def build_pinocchio_model(urdf_abs: Path):
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(urdf_abs))
    data = model.createData()
    return pin, model, data


def fk_world_boxes(
    pin, model, data, q: np.ndarray, local_boxes: Sequence[LinkBox]
) -> List[WorldPoseBox]:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    # map link name -> frame id
    name_to_fid = {f.name: i for i, f in enumerate(model.frames)}
    out: List[WorldPoseBox] = []
    for box in local_boxes:
        if box.name not in name_to_fid:
            continue
        # pinocchio SE3: prefer .homogeneous; avoid np.array(..., dtype=) (breaks some builds)
        oMf = data.oMf[name_to_fid[box.name]]
        T = np.asarray(oMf.homogeneous if hasattr(oMf, "homogeneous") else oMf, dtype=np.float64)
        out.append(WorldPoseBox(
            name=box.name, T_world_link=T,
            aabb_min=box.aabb_min, aabb_max=box.aabb_max))
    return out


def points_in_any_box(points: np.ndarray, boxes: Sequence[WorldPoseBox]) -> np.ndarray:
    """返回 bool mask：True = 在任一膨胀盒内（应剔除）。"""
    n = points.shape[0]
    inside = np.zeros(n, dtype=bool)
    remaining = np.ones(n, dtype=bool)
    for box in boxes:
        if not remaining.any():
            break
        idx = np.flatnonzero(remaining)
        pts = points[idx]
        R = box.T_world_link[:3, :3]
        t = box.T_world_link[:3, 3]
        local = (R.T @ (pts - t).T).T
        hit = np.all((local >= box.aabb_min) & (local <= box.aabb_max), axis=1)
        inside[idx[hit]] = True
        remaining[idx[hit]] = False
    return inside


def load_joints_deg_from_arm(arm_ip: str, arm_port: int) -> List[float]:
    if _add_calibration_path() is None:
        raise SystemExit('找不到 Calibration，无法读臂关节')
    from modules.robot.realman.legacy_arm_controller import ArmController  # type: ignore
    arm = ArmController(arm_ip, arm_port)
    if not arm.connect():
        raise SystemExit(f'连接机械臂失败 {arm_ip}:{arm_port}')
    try:
        ok, joints = arm.get_current_joints()
        if not ok:
            raise SystemExit(f'读关节失败: {joints}')
        return [float(v) for v in joints]
    finally:
        arm.disconnect()


def load_joints_list(path: Path) -> List[List[float]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict):
        if 'joints_deg' in data and isinstance(data['joints_deg'][0], (int, float)):
            return [[float(v) for v in data['joints_deg']]]
        if 'poses' in data:
            rows = []
            for item in data['poses']:
                j = item.get('joints_deg') or item.get('joint') or item.get('q_deg')
                if j is None or len(j) != 7:
                    raise SystemExit(f'坏 joints: {item}')
                rows.append([float(v) for v in j])
            return rows
        if 'samples' in data:
            return [[float(v) for v in s['joints_deg']] for s in data['samples']]
    if isinstance(data, list):
        if data and isinstance(data[0], (int, float)):
            return [[float(v) for v in data]]
        return [[float(v) for v in row] for row in data]
    raise SystemExit(f'无法解析 joints JSON: {path}')


def filter_ply(ply_in: Path, ply_out: Path, inside_mask: np.ndarray) -> Tuple[int, int]:
    import trimesh
    mesh = trimesh.load(str(ply_in), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(
            g for g in mesh.geometry.values()
            if isinstance(g, trimesh.Trimesh)))
    keep = ~inside_mask
    n0 = len(mesh.vertices)
    # 顶点过滤：用 mask 重建
    if hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces):
        # 丢掉含被删顶点的面
        vkeep = keep
        fmask = vkeep[mesh.faces].all(axis=1)
        mesh.update_faces(fmask)
        mesh.remove_unreferenced_vertices()
    else:
        mesh.vertices = mesh.vertices[keep]
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            try:
                mesh.visual.vertex_colors = mesh.visual.vertex_colors[keep]
            except Exception:
                pass
    n1 = len(mesh.vertices)
    mesh.export(str(ply_out))
    return n0, n1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Experiment G: URDF inflated-box self filter')
    p.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    p.add_argument('--ply', type=Path, default=DEFAULT_PLY)
    p.add_argument('--out', type=Path, default=ROOT / 'live_mesh_self_filtered.ply')
    p.add_argument('--inflate-m', type=float, default=0.05,
                   help='每个连杆 AABB 每侧膨胀量（米），默认 5cm')
    p.add_argument('--from-arm', action='store_true', help='从 RealMan 读当前 J1～J7（度）')
    p.add_argument('--arm-ip', default='192.168.1.18')
    p.add_argument('--arm-port', type=int, default=8080)
    p.add_argument('--joints-deg', type=str, default='',
                   help='逗号分隔 7 个数（度），如 0,10,0,20,0,30,0')
    p.add_argument('--joints-json', type=Path, default=None,
                   help='多组关节 JSON（扫视并集剔除）')
    p.add_argument('--dump-boxes-json', type=Path, default=None,
                   help='可选：写出膨胀盒（调试）')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.urdf.is_file():
        raise SystemExit(f'URDF 不存在: {args.urdf}')
    if not args.ply.is_file():
        raise SystemExit(f'PLY 不存在: {args.ply}')

    joints_list: List[List[float]] = []
    if args.joints_json is not None:
        joints_list = load_joints_list(args.joints_json)
    elif args.joints_deg.strip():
        joints_list = [[float(x) for x in args.joints_deg.split(',')]]
        if len(joints_list[0]) != 7:
            raise SystemExit('--joints-deg 需要 7 个数')
    elif args.from_arm:
        joints_list = [load_joints_deg_from_arm(args.arm_ip, args.arm_port)]
    else:
        raise SystemExit('请指定 --from-arm / --joints-deg / --joints-json')

    tmp_urdf = Path('/tmp') / f'{args.urdf.stem}_abs.urdf'
    urdf_abs = prepare_urdf_abs(args.urdf, tmp_urdf)
    local_boxes = load_link_boxes_local(urdf_abs, args.inflate_m)
    if not local_boxes:
        raise SystemExit('URDF 未解析到任何 collision mesh')
    print(f'连杆盒 {len(local_boxes)} 个，inflate={args.inflate_m*100:.1f} cm/侧')
    for b in local_boxes:
        size = b.aabb_max - b.aabb_min
        print(f'  {b.name}: size_m={size.round(3).tolist()}')

    pin, model, data = build_pinocchio_model(urdf_abs)
    if model.nq != 7:
        raise SystemExit(f'期望 7DoF，got nq={model.nq}')

    all_world: List[WorldPoseBox] = []
    for ji, jdeg in enumerate(joints_list):
        q = np.deg2rad(np.asarray(jdeg, dtype=np.float64))
        boxes = fk_world_boxes(pin, model, data, q, local_boxes)
        print(f'姿态[{ji}] joints_deg={[round(v, 2) for v in jdeg]} → {len(boxes)} boxes')
        all_world.extend(boxes)

    if args.dump_boxes_json:
        dump = []
        for b in all_world:
            dump.append({
                'link': b.name,
                'T_world_link': b.T_world_link.tolist(),
                'aabb_min': b.aabb_min.tolist(),
                'aabb_max': b.aabb_max.tolist(),
            })
        args.dump_boxes_json.write_text(
            json.dumps({'inflate_m': args.inflate_m, 'boxes': dump}, indent=2),
            encoding='utf-8')
        print(f'已写盒子 → {args.dump_boxes_json}')

    import trimesh
    mesh = trimesh.load(str(args.ply), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(
            g for g in mesh.geometry.values()
            if isinstance(g, trimesh.Trimesh)))
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    inside = points_in_any_box(pts, all_world)
    n_in = int(inside.sum())
    print(f'顶点 {len(pts)}，盒内剔除 {n_in} ({100.0*n_in/max(len(pts),1):.1f}%)')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n0, n1 = filter_ply(args.ply, args.out, inside)
    print(f'输出 {args.out}  顶点 {n0} → {n1}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
