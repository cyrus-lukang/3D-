#!/usr/bin/env python3
"""把 URDF + 地图 PLY + 关节序列 打成 Foxglove 可打开的 MCAP（JSON schema）。

默认档位：接近完整顶点 + 顶点色（单帧 ~10–20MB；扫 5 分钟每 2s → 约 1.5–3GB）。

依赖（主机执行一次）：
  pip3 install 'mcap>=1.1.0' trimesh

静态地图：
  python3 tools/record_foxglove_mcap.py \\
    --ply live_mesh_self_filtered.ply \\
    --joints-json gpu_logs/scan_joints.json \\
    --out gpu_logs/session_demo.mcap

渐进彩色地图（mapper --progressive_map_dir 产出）：
  python3 tools/record_foxglove_mcap.py \\
    --progressive-dir gpu_logs/progressive_map \\
    --joints-json gpu_logs/scan_joints.json \\
    --out gpu_logs/session_progressive.mcap

Foxglove：Open local file → 打开 mcap → 3D 面板开 /map 与 Frame transforms；
再在 3D → URDF 层加载 RM75-B.urdf，Control mode = Transforms。
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / 'rimbot-TJ-urdf_260822-6/urdf/rimbot-TJ-urdf_260822-6.urdf'
DEFAULT_PLY = ROOT / 'samples/live_mesh_self_filtered.ply'


def _require_mcap():
    try:
        from mcap.writer import Writer  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "缺少 Python 包 mcap。请先执行：\n"
            "  pip3 install 'mcap>=1.1.0'\n"
            f"原始错误: {exc}"
        ) from exc


def load_joints_list(path: Path) -> Tuple[List[List[float]], List[Optional[float]]]:
    """Return (joints_deg rows, grasp_gap per row; None=no pinch overlay)."""
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict) and 'poses' in data:
        rows: List[List[float]] = []
        gaps: List[Optional[float]] = []
        for item in data['poses']:
            j = item.get('joints_deg') or item.get('joint') or item.get('q_deg')
            if j is None or len(j) not in (7, 14):
                raise SystemExit(f'坏 joints (期望 7 或 14): {item}')
            rows.append([float(v) for v in j])
            g = item.get('grasp_gap', None)
            gaps.append(None if g is None else float(g))
        return rows, gaps
    if isinstance(data, dict) and 'joints_deg' in data:
        j = data['joints_deg']
        if j and isinstance(j[0], (int, float)):
            return [[float(v) for v in j]], [None]
    if isinstance(data, list):
        if data and isinstance(data[0], (int, float)):
            return [[float(v) for v in data]], [None]
        return [[float(v) for v in row] for row in data], [None] * len(data)
    raise SystemExit(f'无法解析: {path}')


def prepare_urdf_abs(urdf_path: Path, out_path: Path) -> Path:
    text = urdf_path.read_text(encoding='utf-8')
    pkg_root = urdf_path.resolve().parents[1]
    # package://<any_pkg>/ → absolute package root (parent of urdf/)
    text = re.sub(r'package://[^/]+/', str(pkg_root) + '/', text)
    out_path.write_text(text, encoding='utf-8')
    return out_path


def discover_link_meshes(
    urdf_path: Path, mode: str = 'core'
) -> Dict[str, Path]:
    """Map link_name → STL path from URDF visuals.

    mode=core: base + armL/R link1..7 (+FL/Sensor) + 夹爪/TCP/ToolFL
    mode=full: every visual mesh in the URDF
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()
    pkg_root = urdf_path.resolve().parents[1]
    mesh_dir = pkg_root / 'meshes'
    out: Dict[str, Path] = {}
    for link in root.findall('link'):
        name = link.get('name') or ''
        visual = link.find('visual')
        if visual is None:
            continue
        mesh = visual.find('geometry/mesh')
        if mesh is None:
            continue
        fn = mesh.get('filename') or ''
        stem = Path(fn).name
        path = mesh_dir / stem
        if not path.is_file():
            # already absolute after prepare
            cand = Path(fn)
            if cand.is_file():
                path = cand
            else:
                continue
        out[name] = path

    if mode == 'full':
        return out

    keep = {'base_link'}
    for side in ('armL', 'armR'):
        for i in range(1, 8):
            keep.add(f'{side}_link{i}')
        keep.add(f'{side}_linkFL')
        keep.add(f'{side}_linkSensor')
    # URDF 拼写为 amrL/amrR（非 arm）
    for side in ('amrL', 'amrR', 'armL', 'armR'):
        keep.add(f'{side}_linkToolFL')
        keep.add(f'{side}_linkGripper')
        keep.add(f'{side}_linkTCP')
    return {k: v for k, v in out.items() if k in keep}


def joints_deg_to_q(
    model, jdeg: Sequence[float], drive_arm: str
) -> np.ndarray:
    """Map logged joints → pinocchio q (nq).

    - len==nq: use as-is
    - len==7 and nq==7: single arm
    - len==7 and nq==14: drive armL or armR (other arm zeros)
    """
    q = np.zeros(model.nq, dtype=np.float64)
    j = np.deg2rad(np.asarray(jdeg, dtype=np.float64))
    if len(j) == model.nq:
        return j
    if model.nq == 7 and len(j) == 7:
        return j
    if model.nq == 14 and len(j) == 7:
        # pinocchio joint order: armL_1..7, armR_1..7
        if drive_arm == 'armL':
            q[0:7] = j
        elif drive_arm == 'armR':
            q[7:14] = j
        elif drive_arm == 'both':
            q[0:7] = j
            q[7:14] = j
        else:
            raise SystemExit(f'未知 --drive-arm: {drive_arm}')
        return q
    raise SystemExit(
        f'关节维度不匹配: joints={len(j)} model.nq={model.nq} '
        f'(支持 7→单臂, 7→双臂指定 drive-arm, 或 14→双臂全量)')

def mat_to_pose_quat(T: np.ndarray) -> Dict[str, Any]:
    """4x4 → foxglove Pose (position + orientation xyzw)."""
    R = T[:3, :3]
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (
            R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (
            R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, (
            0.25 * s), (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (
            R[1, 2] + R[2, 1]) / s, 0.25 * s
    return {
        'position': {'x': float(T[0, 3]), 'y': float(T[1, 3]), 'z': float(T[2, 3])},
        'orientation': {
            'x': float(qx), 'y': float(qy), 'z': float(qz), 'w': float(qw)},
    }


def ts_dict(sec: int, nsec: int) -> Dict[str, int]:
    return {'sec': int(sec), 'nsec': int(nsec)}


def load_ply_xyzrgb(ply: Path, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (N,3) float32 xyz and (N,4) uint8 rgba. No trimesh dependency."""
    raw = ply.read_bytes()
    header_end = raw.find(b'end_header\n')
    if header_end < 0:
        raise SystemExit(f'bad ply header: {ply}')
    header = raw[:header_end].decode('ascii', errors='ignore')
    n = None
    for line in header.splitlines():
        if line.startswith('element vertex'):
            n = int(line.split()[-1])
    if not n:
        raise SystemExit(f'no vertex count in {ply}')
    body = raw[header_end + len(b'end_header\n') :]
    # live_cloud.ply: double xyz + uchar rgb (27 bytes/vertex)
    if len(body) < n * 27:
        raise SystemExit(f'ply truncated: {ply}')
    idx = np.arange(n)
    if max_points > 0 and n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
    dt = np.dtype(
        [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
    )
    arr = np.frombuffer(body, dtype=dt, count=n)[idx]
    pts = np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32)
    colors = np.column_stack(
        [arr['r'], arr['g'], arr['b'], np.full(len(arr), 255, dtype=np.uint8)]
    ).astype(np.uint8)
    return pts, colors


def pointcloud_msg_colored(
    pts: np.ndarray, colors: np.ndarray, frame_id: str, sec: int, nsec: int
) -> Dict[str, Any]:
    """Foxglove PointCloud: xyz float32 + rgba uint8, stride 16."""
    n = len(pts)
    buf = np.zeros((n, 16), dtype=np.uint8)
    buf[:, 0:12] = np.ascontiguousarray(pts, dtype=np.float32).view(np.uint8).reshape(n, 12)
    rgba = np.ascontiguousarray(colors[:, :4], dtype=np.uint8)
    buf[:, 12:16] = rgba
    return {
        'timestamp': ts_dict(sec, nsec),
        'frame_id': frame_id,
        'pose': {
            'position': {'x': 0, 'y': 0, 'z': 0},
            'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1},
        },
        'point_stride': 16,
        'fields': [
            {'name': 'x', 'offset': 0, 'type': 7},       # FLOAT32
            {'name': 'y', 'offset': 4, 'type': 7},
            {'name': 'z', 'offset': 8, 'type': 7},
            {'name': 'red', 'offset': 12, 'type': 1},    # UINT8
            {'name': 'green', 'offset': 13, 'type': 1},
            {'name': 'blue', 'offset': 14, 'type': 1},
            {'name': 'alpha', 'offset': 15, 'type': 1},
        ],
        'data': base64.b64encode(buf.tobytes()).decode('ascii'),
    }


def frame_transforms_msg(
    transforms: Sequence[Dict[str, Any]], sec: int, nsec: int
) -> Dict[str, Any]:
    return {'transforms': transforms}


def one_tf(parent: str, child: str, T: np.ndarray, sec: int, nsec: int) -> Dict[str, Any]:
    pose = mat_to_pose_quat(T)
    return {
        'timestamp': ts_dict(sec, nsec),
        'parent_frame_id': parent,
        'child_frame_id': child,
        'translation': pose['position'],
        'rotation': pose['orientation'],
    }


def load_link_stl_bytes(link_paths: Dict[str, Path]) -> Dict[str, bytes]:
    """连杆名 → STL 二进制（顶点留在各自连杆坐标系）。"""
    out: Dict[str, bytes] = {}
    for name, path in link_paths.items():
        if not path.is_file():
            raise SystemExit(f'缺少连杆网格: {path}')
        out[name] = path.read_bytes()
    return out


def _scene_schema_bytes() -> bytes:
    path = Path(__file__).resolve().parent / 'foxglove_SceneUpdate.json'
    if not path.is_file():
        raise SystemExit(f'缺少 {path}')
    return path.read_bytes()


def stl_triangle_points(data: bytes) -> List[Dict[str, float]]:
    """二进制 STL → 连杆坐标系里的三角顶点。不用 STL 模型，避免 Foxglove 再转 90 度。"""
    ntri = struct.unpack_from('<I', data, 80)[0]
    pts: List[Dict[str, float]] = []
    off = 84
    for _ in range(ntri):
        for k in range(3):
            x, y, z = struct.unpack_from('<3f', data, off + 12 + 12 * k)
            pts.append({'x': round(x, 5), 'y': round(y, 5), 'z': round(z, 5)})
        off += 50
    return pts


def arm_scene_msg(meshes: Dict[str, bytes], sec: int, nsec: int) -> Dict[str, Any]:
    """每节挂在同名坐标上，三角面用连杆自身的朝向。"""
    identity = {
        'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
    }
    entities = []
    for name, raw in meshes.items():
        entities.append({
            'timestamp': ts_dict(sec, nsec),
            'frame_id': name,
            'id': name,
            'lifetime': {'sec': 0, 'nsec': 0},
            'frame_locked': True,
            'metadata': [],
            'arrows': [],
            'cubes': [],
            'spheres': [],
            'cylinders': [],
            'lines': [],
            'triangles': [{
                'pose': identity,
                'points': stl_triangle_points(raw),
                'color': {'r': 0.75, 'g': 0.78, 'b': 0.82, 'a': 1.0},
                'colors': [],
                'indices': [],
            }],
            'texts': [],
            'models': [],
        })
    return {'deletions': [], 'entities': entities}


def gripper_pinch_scene_msg(
    tcp_frame: str,
    gap_half: Optional[float],
    sec: int,
    nsec: int,
    tcp_T: Optional[np.ndarray] = None,
    base_frame: str = 'base_link',
) -> Dict[str, Any]:
    """示意指尖合拢。优先挂在 base_link（世界位姿），避免 TCP 坐标系未勾选时看不见。"""
    if gap_half is None:
        return {
            'deletions': [{
                'timestamp': ts_dict(sec, nsec),
                'type': 1,  # MATCHING_ID
                'id': 'demo_gripper_jaws',
            }],
            'entities': [],
        }
    g = max(0.002, float(gap_half))
    cubes = []
    spheres = []
    if tcp_T is not None:
        # 世界系：TCP 附近两片红块沿局部 Y 开合
        R = tcp_T[:3, :3]
        p = tcp_T[:3, 3]
        tip = p + 0.02 * R[:, 0]
        y = R[:, 1]
        for sign in (1.0, -1.0):
            c = tip + sign * g * y
            pose = mat_to_pose_quat(
                np.vstack([
                    np.hstack([R, c.reshape(3, 1)]),
                    np.array([[0.0, 0.0, 0.0, 1.0]]),
                ])
            )
            cubes.append({
                'pose': pose,
                'size': {'x': 0.08, 'y': 0.022, 'z': 0.04},
                'color': {'r': 1.0, 'g': 0.15, 'b': 0.05, 'a': 1.0},
            })
        spheres.append({
            'pose': mat_to_pose_quat(
                np.vstack([
                    np.hstack([R, tip.reshape(3, 1)]),
                    np.array([[0.0, 0.0, 0.0, 1.0]]),
                ])
            ),
            'size': {'x': 0.035, 'y': 0.035, 'z': 0.035},
            'color': {'r': 1.0, 'g': 0.85, 'b': 0.1, 'a': 1.0},
        })
        frame_id = base_frame
    else:
        for sign in (1.0, -1.0):
            cubes.append({
                'pose': {
                    'position': {'x': 0.02, 'y': sign * g, 'z': 0.0},
                    'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                },
                'size': {'x': 0.08, 'y': 0.022, 'z': 0.04},
                'color': {'r': 1.0, 'g': 0.15, 'b': 0.05, 'a': 1.0},
            })
        spheres.append({
            'pose': {
                'position': {'x': 0.02, 'y': 0.0, 'z': 0.0},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
            },
            'size': {'x': 0.035, 'y': 0.035, 'z': 0.035},
            'color': {'r': 1.0, 'g': 0.85, 'b': 0.1, 'a': 1.0},
        })
        frame_id = tcp_frame
    return {
        'deletions': [],
        'entities': [{
            'timestamp': ts_dict(sec, nsec),
            'frame_id': frame_id,
            'id': 'demo_gripper_jaws',
            'lifetime': {'sec': 0, 'nsec': 0},
            'frame_locked': True,
            'metadata': [{'key': 'grasp_gap', 'value': f'{g:.4f}'}],
            'arrows': [],
            'cubes': cubes,
            'spheres': spheres,
            'cylinders': [],
            'lines': [],
            'triangles': [],
            'texts': [],
            'models': [],
        }],
    }


def height_plane_scene_msg(
    z_m: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    sec: int,
    nsec: int,
    frame_id: str = 'base_link',
) -> Dict[str, Any]:
    """Horizontal reference plane at fixed Z in robot base frame."""
    identity = {
        'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
    }
    # Two triangles = one quad in XY at z=z_m
    c00 = {'x': xmin, 'y': ymin, 'z': z_m}
    c10 = {'x': xmax, 'y': ymin, 'z': z_m}
    c11 = {'x': xmax, 'y': ymax, 'z': z_m}
    c01 = {'x': xmin, 'y': ymax, 'z': z_m}
    pts = [c00, c10, c11, c00, c11, c01]
    return {
        'deletions': [],
        'entities': [{
            'timestamp': ts_dict(sec, nsec),
            'frame_id': frame_id,
            'id': f'height_plane_z_{z_m:.3f}',
            'lifetime': {'sec': 0, 'nsec': 0},
            'frame_locked': True,
            'metadata': [{'key': 'z_m', 'value': f'{z_m}'}],
            'arrows': [],
            'cubes': [],
            'spheres': [],
            'cylinders': [],
            'lines': [],
            'triangles': [{
                'pose': identity,
                'points': pts,
                'color': {'r': 0.2, 'g': 0.55, 'b': 0.95, 'a': 0.35},
                'colors': [],
                'indices': [],
            }],
            'texts': [{
                'pose': {
                    'position': {'x': float(xmin), 'y': float(ymin), 'z': float(z_m) + 0.02},
                    'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                },
                'billboard': True,
                'font_size': 0.06,
                'scale_invariant': False,
                'color': {'r': 0.1, 'g': 0.35, 'b': 0.9, 'a': 1.0},
                'text': f'z={z_m:.3f} m (base_link)',
            }],
            'models': [],
        }],
    }


def load_progressive_manifest(prog_dir: Path) -> List[Dict[str, Any]]:
    man = prog_dir / 'manifest.jsonl'
    if not man.is_file():
        # fallback: sorted frame_*.ply
        frames = sorted(prog_dir.glob('frame_*.ply'))
        if not frames:
            raise SystemExit(f'渐进目录无 manifest.jsonl / frame_*.ply: {prog_dir}')
        return [
            {'idx': i, 'file': f.name, 't_rel_s': float(i) * 2.0, 'vertices': -1}
            for i, f in enumerate(frames)
        ]
    rows = []
    for line in man.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f'manifest 为空: {man}')
    return rows


def _json_time(desc: str) -> Dict[str, Any]:
    return {
        'type': 'object',
        'title': 'time',
        'description': desc,
        'properties': {
            'sec': {'type': 'integer', 'minimum': 0},
            'nsec': {'type': 'integer', 'minimum': 0, 'maximum': 999999999},
        },
        'required': ['sec', 'nsec'],
    }


def _json_vec3(title: str, desc: str) -> Dict[str, Any]:
    return {
        'type': 'object',
        'title': title,
        'description': desc,
        'properties': {
            'x': {'type': 'number'},
            'y': {'type': 'number'},
            'z': {'type': 'number'},
        },
        'required': ['x', 'y', 'z'],
    }


def _json_quat(title: str, desc: str) -> Dict[str, Any]:
    return {
        'type': 'object',
        'title': title,
        'description': desc,
        'properties': {
            'x': {'type': 'number'},
            'y': {'type': 'number'},
            'z': {'type': 'number'},
            'w': {'type': 'number'},
        },
        'required': ['x', 'y', 'z', 'w'],
    }


# 每一层都必须带 type。缺 items / properties 时 Foxglove 会报
# Cannot read properties of undefined (reading 'type')。
# NumericType 用 integer，不用 oneOf：oneOf 分支没有 type，同样会崩。
POINTCLOUD_SCHEMA = json.dumps({
    'title': 'foxglove.PointCloud',
    'description': 'A collection of N-dimensional points',
    '$comment': 'Generated for Foxglove jsonschema; nested types required',
    'type': 'object',
    'properties': {
        'timestamp': _json_time('Timestamp of point cloud'),
        'frame_id': {'type': 'string', 'description': 'Frame of reference'},
        'pose': {
            'type': 'object',
            'title': 'foxglove.Pose',
            'description': 'Origin of the point cloud relative to frame_id',
            'properties': {
                'position': _json_vec3('foxglove.Vector3', 'Position'),
                'orientation': _json_quat('foxglove.Quaternion', 'Orientation'),
            },
            'required': ['position', 'orientation'],
        },
        'point_stride': {'type': 'integer', 'minimum': 0},
        'fields': {
            'type': 'array',
            'items': {
                'type': 'object',
                'title': 'foxglove.PackedElementField',
                'properties': {
                    'name': {'type': 'string'},
                    'offset': {'type': 'integer', 'minimum': 0},
                    'type': {
                        'type': 'integer',
                        'minimum': 0,
                        'maximum': 8,
                        'description': '1=UINT8 7=FLOAT32',
                    },
                },
                'required': ['name', 'offset', 'type'],
            },
        },
        'data': {'type': 'string', 'contentEncoding': 'base64'},
    },
    'required': [
        'timestamp', 'frame_id', 'pose', 'point_stride', 'fields', 'data'],
})

FRAME_TRANSFORMS_SCHEMA = json.dumps({
    'title': 'foxglove.FrameTransforms',
    'description': 'An array of FrameTransform messages',
    'type': 'object',
    'properties': {
        'transforms': {
            'type': 'array',
            'items': {
                'type': 'object',
                'title': 'foxglove.FrameTransform',
                'properties': {
                    'timestamp': _json_time('Timestamp of transform'),
                    'parent_frame_id': {'type': 'string'},
                    'child_frame_id': {'type': 'string'},
                    'translation': _json_vec3(
                        'foxglove.Vector3', 'Child origin in the parent frame'),
                    'rotation': _json_quat(
                        'foxglove.Quaternion', 'Child orientation in the parent frame'),
                },
                'required': [
                    'timestamp', 'parent_frame_id', 'child_frame_id',
                    'translation', 'rotation'],
            },
        },
    },
    'required': ['transforms'],
})

STRING_SCHEMA = json.dumps({
    'title': 'foxglove.RawMessage',
    'type': 'object',
    'properties': {'data': {'type': 'string'}},
})


def _sec_nsec(t_sec: float) -> Tuple[int, int]:
    sec = int(t_sec)
    nsec = int(round((t_sec - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec += 1
        nsec -= 1_000_000_000
    return sec, nsec


def _apply_cloud_offset(pts: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    if abs(dx) < 1e-12 and abs(dy) < 1e-12 and abs(dz) < 1e-12:
        return pts
    out = pts.copy()
    out[:, 0] += dx
    out[:, 1] += dy
    out[:, 2] += dz
    return out


def main() -> int:
    _require_mcap()
    from mcap.writer import Writer

    p = argparse.ArgumentParser(
        description='Record Foxglove MCAP (colored map + TF + URDF). '
                    'Default: near-full vertices + vertex colors.')
    p.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    p.add_argument('--ply', type=Path, default=None,
                   help='Static colored PLY (default: live_mesh_self_filtered or live_mesh)')
    p.add_argument('--progressive-dir', type=Path, default=None,
                   help='Dir with frame_*.ply + manifest.jsonl from mapper '
                        '--progressive_map_dir (writes growing /map)')
    p.add_argument('--joints-json', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'gpu_logs' / 'session_foxglove.mcap')
    p.add_argument('--dt', type=float, default=1.0,
                   help='相邻记录姿态的间隔秒数。中间按 10Hz 插值，播放时机械臂才会慢慢动')
    p.add_argument('--max-points', type=int, default=0,
                   help='每帧最多点数；0=接近完整不降采样（默认）')
    p.add_argument('--frame-id', default='base_link')
    p.add_argument('--drive-arm',
                   choices=('armR', 'armL', 'both'),
                   default='armR',
                   help='当 joints 只有 7 维而 URDF 是双臂 14 维时，驱动哪一侧（默认右臂 armR）')
    p.add_argument('--mesh-mode',
                   choices=('core', 'full'),
                   default='core',
                   help='core=基座+双臂连杆+夹爪；full=含相机等全部 visual')
    p.add_argument('--height-plane-z',
                   type=float,
                   default=None,
                   help='在 base_link 画一张固定高度水平面（米）。例: 0.0 或 0.75')
    p.add_argument('--height-plane-margin-m',
                   type=float,
                   default=0.15,
                   help='高度面相对点云 XY 外扩（米），默认 0.15')
    p.add_argument('--cloud-z-offset-m',
                   type=float,
                   default=0.0,
                   help='点云在 base_link 下整体沿 Z 平移（米），例: 0.75')
    p.add_argument('--cloud-x-offset-m',
                   type=float,
                   default=0.0,
                   help='点云在 base_link 下整体沿 X 平移（米），例: 0.2')
    p.add_argument('--cloud-y-offset-m',
                   type=float,
                   default=0.0,
                   help='点云在 base_link 下整体沿 Y 平移（米）')
    args = p.parse_args()

    if args.progressive_dir is None and args.ply is None:
        for cand in (ROOT / 'live_mesh_self_filtered.ply', DEFAULT_PLY):
            if cand.is_file():
                args.ply = cand
                break
        if args.ply is None:
            raise SystemExit('缺少 --ply 或 --progressive-dir')

    if args.progressive_dir is not None and not args.progressive_dir.is_dir():
        raise SystemExit(f'渐进目录不存在: {args.progressive_dir}')
    if args.progressive_dir is None and not args.ply.is_file():
        raise SystemExit(f'缺少地图: {args.ply}')
    if not args.joints_json.is_file():
        raise SystemExit(
            f'缺少关节日志: {args.joints_json}\n'
            '请先在扫视时加: --log-joints gpu_logs/scan_joints.json')
    if not args.urdf.is_file():
        raise SystemExit(f'缺少 URDF: {args.urdf}')

    joints_list, grasp_gaps = load_joints_list(args.joints_json)
    if len(grasp_gaps) < len(joints_list):
        grasp_gaps = list(grasp_gaps) + [None] * (len(joints_list) - len(grasp_gaps))
    n_grasp = sum(1 for g in grasp_gaps if g is not None)
    tmp_urdf = Path('/tmp') / f'{args.urdf.stem}_foxglove.urdf'
    urdf_abs = prepare_urdf_abs(args.urdf, tmp_urdf)

    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(urdf_abs))
    data = model.createData()
    print(f'URDF nq={model.nq} drive_arm={args.drive_arm} mesh_mode={args.mesh_mode}')
    if n_grasp:
        print(f'夹合关键帧 {n_grasp}/{len(joints_list)}（将写入 /gripper）')
    link_paths = discover_link_meshes(args.urdf, mode=args.mesh_mode)
    if not link_paths:
        raise SystemExit(f'URDF 未找到可用 visual mesh: {args.urdf}')
    print(f'连杆网格 {len(link_paths)}: {sorted(link_paths)}')
    body_frames = [
        (i, f.name) for i, f in enumerate(model.frames)
        if f.name in link_paths or f.name == 'base_link'
    ]

    prog_rows: Optional[List[Dict[str, Any]]] = None
    static_pts = static_colors = None
    if args.progressive_dir is not None:
        prog_rows = load_progressive_manifest(args.progressive_dir)
        print(f'渐进地图 {len(prog_rows)} 帧 @ {args.progressive_dir}；'
              f'关节 {len(joints_list)}；写出 {args.out}（近完整顶点+颜色）')
    else:
        static_pts, static_colors = load_ply_xyzrgb(args.ply, args.max_points)
        dx, dy, dz = (float(args.cloud_x_offset_m), float(args.cloud_y_offset_m),
                      float(args.cloud_z_offset_m))
        static_pts = _apply_cloud_offset(static_pts, dx, dy, dz)
        if abs(dx) > 1e-12 or abs(dy) > 1e-12 or abs(dz) > 1e-12:
            print(f'点云偏移 base_link  Δ=({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) m')
        print(f'静态彩色地图 {len(static_pts)} 点，关节 {len(joints_list)}，写出 {args.out}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0_ns = time.time_ns()

    with open(args.out, 'wb') as f:
        w = Writer(f)
        w.start(profile='', library='nvblox-foxglove-recorder')
        pc_schema = w.register_schema(
            name='foxglove.PointCloud', encoding='jsonschema',
            data=POINTCLOUD_SCHEMA.encode('utf-8'))
        tf_schema = w.register_schema(
            name='foxglove.FrameTransforms', encoding='jsonschema',
            data=FRAME_TRANSFORMS_SCHEMA.encode('utf-8'))
        desc_schema = w.register_schema(
            name='foxglove.RawMessage', encoding='jsonschema',
            data=STRING_SCHEMA.encode('utf-8'))
        scene_schema = w.register_schema(
            name='foxglove.SceneUpdate', encoding='jsonschema',
            data=_scene_schema_bytes())

        ch_map = w.register_channel(
            topic='/map', message_encoding='json', schema_id=pc_schema)
        ch_tf = w.register_channel(
            topic='/tf', message_encoding='json', schema_id=tf_schema)
        ch_desc = w.register_channel(
            topic='/robot_description', message_encoding='json',
            schema_id=desc_schema)
        ch_arm = w.register_channel(
            topic='/arm', message_encoding='json', schema_id=scene_schema)
        ch_gripper = w.register_channel(
            topic='/gripper', message_encoding='json', schema_id=scene_schema)
        ch_height = w.register_channel(
            topic='/height_plane', message_encoding='json', schema_id=scene_schema)

        meshes = load_link_stl_bytes(link_paths)
        tcp_frame = (
            'amrL_linkTCP' if args.drive_arm == 'armL'
            else 'amrR_linkTCP' if args.drive_arm == 'armR'
            else 'amrL_linkTCP'
        )

        # t=0: urdf text once
        sec, nsec = 0, 0
        log_time = t0_ns
        w.add_message(
            channel_id=ch_desc, log_time=log_time, publish_time=log_time,
            data=json.dumps({
                'data': args.urdf.read_text(encoding='utf-8'),
                'note': 'Paste/load the same URDF in Foxglove 3D URDF layer; '
                        'mesh paths need desktop file:// access.',
                'urdf_abs_path': str(urdf_abs),
            }).encode())

        if prog_rows is None:
            w.add_message(
                channel_id=ch_map, log_time=log_time, publish_time=log_time,
                data=json.dumps(pointcloud_msg_colored(
                    static_pts, static_colors, args.frame_id, sec, nsec)).encode())
            ref_pts = static_pts
        else:
            ref_pts = None
            # growing colored map on progressive timestamps
            for rec in prog_rows:
                ply_path = args.progressive_dir / rec['file']
                if not ply_path.is_file():
                    print(f'  skip missing {ply_path}')
                    continue
                t_sec = float(rec.get('t_rel_s', 0.0))
                sec, nsec = _sec_nsec(t_sec)
                log_time = t0_ns + int(t_sec * 1e9)
                pts, colors = load_ply_xyzrgb(ply_path, args.max_points)
                pts = _apply_cloud_offset(
                    pts,
                    float(args.cloud_x_offset_m),
                    float(args.cloud_y_offset_m),
                    float(args.cloud_z_offset_m),
                )
                if ref_pts is None:
                    ref_pts = pts
                w.add_message(
                    channel_id=ch_map, log_time=log_time, publish_time=log_time,
                    data=json.dumps(pointcloud_msg_colored(
                        pts, colors, args.frame_id, sec, nsec)).encode())
                print(f'  map frame {rec.get("idx", "?")} t={t_sec:.2f}s '
                      f'pts={len(pts)} file={rec["file"]}')

        # Fixed-height reference plane in base_link (optional)
        if args.height_plane_z is not None:
            if ref_pts is None or len(ref_pts) == 0:
                raise SystemExit('--height-plane-z 需要有效点云以确定平面范围')
            m = float(args.height_plane_margin_m)
            xmin = float(ref_pts[:, 0].min() - m)
            xmax = float(ref_pts[:, 0].max() + m)
            ymin = float(ref_pts[:, 1].min() - m)
            ymax = float(ref_pts[:, 1].max() + m)
            z0 = float(args.height_plane_z)
            sec, nsec = 0, 0
            w.add_message(
                channel_id=ch_height, log_time=t0_ns, publish_time=t0_ns,
                data=json.dumps(height_plane_scene_msg(
                    z0, xmin, xmax, ymin, ymax, sec, nsec, args.frame_id)).encode())
            print(f'高度面 /height_plane  z={z0:.3f} m  '
                  f'XY=[{xmin:.2f},{xmax:.2f}]x[{ymin:.2f},{ymax:.2f}] (base_link)')
        nj = len(joints_list[0])
        samples = []
        if len(joints_list) == 1:
            samples = [(0.0, joints_list[0], grasp_gaps[0])]
        else:
            for ia, (a, b) in enumerate(zip(joints_list, joints_list[1:])):
                ga = grasp_gaps[ia] if ia < len(grasp_gaps) else None
                gb = grasp_gaps[ia + 1] if ia + 1 < len(grasp_gaps) else None
                n = max(1, int(round(args.dt * 10)))
                for k in range(n):
                    u = k / n
                    t = len(samples) / 10.0
                    q = [(1 - u) * a[i] + u * b[i] for i in range(nj)]
                    if ga is None and gb is None:
                        g = None
                    elif ga is None:
                        g = gb
                    elif gb is None:
                        g = ga
                    else:
                        g = (1.0 - u) * ga + u * gb
                    samples.append((t, q, g))
            samples.append(
                (len(samples) / 10.0, joints_list[-1], grasp_gaps[-1])
            )
        if prog_rows:
            duration = float(prog_rows[-1].get('t_rel_s') or 0.0)
            end = samples[-1][0] if samples else 0.0
            if duration > end > 0:
                scale = duration / end
                samples = [(t * scale, q, g) for t, q, g in samples]

        last_g_shown = False
        for i, (t_sec, jdeg, gap) in enumerate(samples):
            sec, nsec = _sec_nsec(t_sec)
            log_time = t0_ns + int(t_sec * 1e9)
            q = joints_deg_to_q(model, jdeg, args.drive_arm)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            transforms = [one_tf('map', args.frame_id, np.eye(4), sec, nsec)]
            for fid, name in body_frames:
                T = np.array(data.oMf[fid], dtype=np.float64)
                if name == 'base_link':
                    continue
                transforms.append(one_tf(args.frame_id, name, T, sec, nsec))
            w.add_message(
                channel_id=ch_tf, log_time=log_time, publish_time=log_time,
                data=json.dumps(frame_transforms_msg(transforms, sec, nsec)).encode())
            if i == 0:
                w.add_message(
                    channel_id=ch_arm, log_time=log_time, publish_time=log_time,
                    data=json.dumps(arm_scene_msg(meshes, sec, nsec)).encode())
            if gap is not None or last_g_shown:
                # TCP 世界位姿（挂 base_link，不依赖是否勾 TCP 子坐标系）
                tcp_T = None
                try:
                    fid_tcp = model.getFrameId(tcp_frame)
                    tcp_T = np.array(data.oMf[fid_tcp], dtype=np.float64)
                except Exception:
                    tcp_T = None
                grip_msg = gripper_pinch_scene_msg(
                    tcp_frame, gap, sec, nsec, tcp_T=tcp_T, base_frame=args.frame_id
                )
                raw = json.dumps(grip_msg).encode()
                # 同时写 /gripper 与 /arm，避免只勾了 /arm 时看不见
                w.add_message(
                    channel_id=ch_gripper,
                    log_time=log_time,
                    publish_time=log_time,
                    data=raw,
                )
                w.add_message(
                    channel_id=ch_arm,
                    log_time=log_time,
                    publish_time=log_time,
                    data=raw,
                )
                last_g_shown = gap is not None
            if i == 0 or i + 1 == len(samples) or i % 10 == 0:
                print(f'  tf {i+1}/{len(samples)} t={t_sec:.2f}s '
                      f'joints={[round(v,1) for v in jdeg]}')
        w.finish()

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f'完成 → {args.out} ({size_mb:.1f} MB)')
    print('Foxglove：3D 勾 /map（Circle + RGBA）、/arm、/gripper；可选 /height_plane。不要勾 /tf。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
