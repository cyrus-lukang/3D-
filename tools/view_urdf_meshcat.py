#!/usr/bin/env python3
"""Meshcat URDF + point cloud: click two targets, arm IK-moves there."""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import meshcat.geometry as mcg
import numpy as np
import pinocchio as pin
import tornado.ioloop
import tornado.web
import tornado.websocket
from meshcat.servers.zmqserver import VIEWER_ROOT
from pinocchio.visualize import MeshcatVisualizer
from tornado.websocket import websocket_connect

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / 'rimbot-TJ-urdf_260822-6/urdf/rimbot-TJ-urdf_260822-6.urdf'
DEFAULT_PLY = REPO_ROOT / 'samples/live_mesh_self_filtered.ply'
HISTORY_ROOT = REPO_ROOT / 'traj_history'
FOXGLOVE_OUT = REPO_ROOT / 'samples'

ARM_FRAMES = {
    'L': 'amrL_linkTCP',
    'R': 'amrR_linkTCP',
}
ARM_FALLBACK = {
    'L': 'armL_linkFL',
    'R': 'armR_linkFL',
}


def load_ply_xyzrgb(path: Path, max_points: int = 200_000):
    raw = path.read_bytes()
    header_end = raw.find(b'end_header\n')
    if header_end < 0:
        raise SystemExit(f'bad ply header: {path}')
    header = raw[:header_end].decode('ascii', errors='ignore')
    n = None
    for line in header.splitlines():
        if line.startswith('element vertex'):
            n = int(line.split()[-1])
    if not n:
        raise SystemExit(f'no vertex count in {path}')
    body = raw[header_end + len(b'end_header\n') :]
    if len(body) < n * 27:
        raise SystemExit(f'ply truncated: {path}')
    idx = np.arange(n)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
    dt = np.dtype(
        [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
    )
    arr = np.frombuffer(body, dtype=dt, count=n)[idx]
    xyz = np.vstack([arr['x'], arr['y'], arr['z']]).astype(np.float64)
    rgb = np.vstack([arr['r'], arr['g'], arr['b']]).astype(np.float64) / 255.0
    return xyz, rgb


def prepare_urdf_abs(urdf_path: Path, out_path: Path) -> Path:
    text = urdf_path.read_text(encoding='utf-8')
    pkg_root = urdf_path.resolve().parents[1]
    text = re.sub(r'package://[^/]+/', str(pkg_root) + '/', text)
    out_path.write_text(text, encoding='utf-8')
    return out_path


def joint_meta(model: pin.Model):
    meta = []
    for jid in range(1, model.njoints):
        jmodel = model.joints[jid]
        if jmodel.nq < 1:
            continue
        idx = jmodel.idx_q
        lo = float(model.lowerPositionLimit[idx]) if idx < model.nq else -np.pi
        hi = float(model.upperPositionLimit[idx]) if idx < model.nq else np.pi
        if not np.isfinite(lo):
            lo = -np.pi
        if not np.isfinite(hi):
            hi = np.pi
        if abs(hi - lo) < 1e-9:
            lo, hi = -np.pi, np.pi
        meta.append(
            {
                'name': model.names[jid],
                'idx_q': int(idx),
                'nq': int(jmodel.nq),
                'lo': lo,
                'hi': hi,
            }
        )
    return meta


def resolve_frame(model: pin.Model, arm: str) -> int:
    for name in (ARM_FRAMES[arm], ARM_FALLBACK[arm]):
        try:
            return model.getFrameId(name)
        except Exception:
            continue
    raise SystemExit(f'no EE frame for arm {arm}')


def arm_q_indices(model: pin.Model, arm: str) -> list[int]:
    prefix = 'armL_' if arm == 'L' else 'armR_'
    # right joints may be armR_Joint*
    idxs = []
    for jid in range(1, model.njoints):
        name = model.names[jid]
        if arm == 'L' and name.startswith('armL_'):
            idxs.append(model.joints[jid].idx_q)
        elif arm == 'R' and (name.startswith('armR_') or name.startswith('armR_J')):
            idxs.append(model.joints[jid].idx_q)
    return idxs[:7]


def clamp_q(model: pin.Model, q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for i in range(model.nq):
        lo = model.lowerPositionLimit[i]
        hi = model.upperPositionLimit[i]
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo + 1e-9:
            out[i] = np.clip(out[i], lo, hi)
    return out


def _is_kin_close(model: pin.Model, j1: int, j2: int, max_hops: int = 2) -> bool:
    """True if one joint is within max_hops ancestors of the other."""
    if j1 == j2:
        return True

    def within(src: int, dst: int) -> bool:
        cur = src
        for _ in range(max_hops):
            cur = model.parents[cur]
            if cur == dst:
                return True
            if cur == 0:
                break
        return False

    return within(j1, j2) or within(j2, j1)


def build_self_collision_model(model: pin.Model, collision_model: pin.GeometryModel) -> pin.GeometryModel:
    """Clone-ish setup: add pairs and drop near-adjacent links."""
    collision_model.addAllCollisionPairs()
    remove = []
    for i, p in enumerate(collision_model.collisionPairs):
        j1 = collision_model.geometryObjects[p.first].parentJoint
        j2 = collision_model.geometryObjects[p.second].parentJoint
        if _is_kin_close(model, j1, j2, max_hops=2):
            remove.append(i)
    for i in reversed(remove):
        collision_model.removeCollisionPair(collision_model.collisionPairs[i])
    return collision_model


def geom_sphere_world(collision_model: pin.GeometryModel, cdata: pin.GeometryData, gid: int):
    """Approximate collision geom as a world-frame sphere (center, radius)."""
    go = collision_model.geometryObjects[gid]
    oMg = cdata.oMg[gid]
    cg = go.geometry
    try:
        cg.computeLocalAABB()
        c_local = np.asarray(cg.aabb_local.center(), dtype=float).reshape(3)
        r = float(getattr(cg, 'aabb_radius', 0.0))
        if not np.isfinite(r) or r <= 1e-6:
            sizes = np.asarray(cg.aabb_local.max_, dtype=float) - np.asarray(
                cg.aabb_local.min_, dtype=float
            )
            r = 0.5 * float(np.linalg.norm(sizes))
    except Exception:
        c_local = np.zeros(3)
        r = 0.05
    if not np.isfinite(r) or r > 2.0:
        # insane AABB → fallback small capsule-ish radius
        r = 0.08
    c_world = np.asarray(oMg.act(c_local), dtype=float).reshape(3)
    return c_world, float(max(r, 0.015))


def _geom_name_is_arm(name: str, arm: str) -> bool:
    n = name.lower()
    if arm == 'L':
        return 'arml' in n or 'amrl' in n
    return 'armr' in n or 'amrr' in n


class CollisionChecker:
    def __init__(
        self,
        model: pin.Model,
        data: pin.Data,
        collision_model: pin.GeometryModel | None,
        cloud_xyz: np.ndarray | None,
        inflate_self_m: float = 0.02,
        inflate_cloud_m: float = 0.03,
        enable_self: bool = True,
        enable_cloud: bool = True,
    ):
        from scipy.spatial import cKDTree

        self.model = model
        self.data = data
        self.collision_model = collision_model
        self.cdata = pin.GeometryData(collision_model) if collision_model is not None else None
        self.cloud_xyz = cloud_xyz
        self.tree = None
        if cloud_xyz is not None and cloud_xyz.shape[1] > 0:
            # 规划用稀疏点云，大幅加速 query
            pts = cloud_xyz.T
            if pts.shape[0] > 20000:
                idx = np.linspace(0, pts.shape[0] - 1, 20000).astype(np.int64)
                pts = pts[idx]
            self.tree = cKDTree(pts)
        self.inflate_self_m = float(inflate_self_m)
        self.inflate_cloud_m = float(inflate_cloud_m)
        self.enable_self = enable_self
        self.enable_cloud = enable_cloud
        self.active_arm = 'L'
        self._arm_geom_ids: dict[str, list[int]] = {'L': [], 'R': []}
        if collision_model is not None:
            for gid in range(collision_model.ngeoms):
                name = collision_model.geometryObjects[gid].name
                if _geom_name_is_arm(name, 'L'):
                    self._arm_geom_ids['L'].append(gid)
                if _geom_name_is_arm(name, 'R'):
                    self._arm_geom_ids['R'].append(gid)
        # 规划阶段可关掉自碰以提速
        self.fast_mode = False

    def set_params(
        self,
        inflate_self_m: float | None = None,
        inflate_cloud_m: float | None = None,
        enable_self: bool | None = None,
        enable_cloud: bool | None = None,
        active_arm: str | None = None,
    ):
        if inflate_self_m is not None:
            self.inflate_self_m = float(inflate_self_m)
        if inflate_cloud_m is not None:
            self.inflate_cloud_m = float(inflate_cloud_m)
        if enable_self is not None:
            self.enable_self = bool(enable_self)
        if enable_cloud is not None:
            self.enable_cloud = bool(enable_cloud)
        if active_arm is not None:
            self.active_arm = active_arm

    def check(self, q: np.ndarray) -> tuple[bool, str]:
        """Return (ok, reason). ok=True means collision-free."""
        do_self = (
            self.enable_self
            and not self.fast_mode
            and self.collision_model is not None
            and self.cdata is not None
            and len(self.collision_model.collisionPairs) > 0
        )
        do_cloud = (
            self.enable_cloud
            and self.tree is not None
            and self.collision_model is not None
            and self.cdata is not None
        )

        if do_self:
            for i in range(len(self.cdata.collisionRequests)):
                self.cdata.collisionRequests[i].security_margin = self.inflate_self_m
            # stop_at_first=True：撞到一对就返回，快很多
            pin.computeCollisions(
                self.model, self.data, self.collision_model, self.cdata, q, True
            )
            for i, r in enumerate(self.cdata.collisionResults):
                if r.isCollision():
                    p = self.collision_model.collisionPairs[i]
                    n1 = self.collision_model.geometryObjects[p.first].name
                    n2 = self.collision_model.geometryObjects[p.second].name
                    return False, f'self: {n1} ↔ {n2}'
        else:
            pin.forwardKinematics(self.model, self.data, q)

        if do_cloud:
            pin.updateGeometryPlacements(
                self.model, self.data, self.collision_model, self.cdata
            )
            for gid in self._arm_geom_ids.get(self.active_arm, []):
                name = self.collision_model.geometryObjects[gid].name
                center, radius = geom_sphere_world(self.collision_model, self.cdata, gid)
                # 最近邻距离，比 query_ball 更稳更快
                d, _ = self.tree.query(center, k=1)
                if float(d) <= radius + self.inflate_cloud_m:
                    return False, f'cloud: {name}'
        return True, 'ok'


def R_gripper_down(yaw_rad: float = 0.0) -> np.ndarray:
    """Gripper frame: +X (tool long axis) along world -Z."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
    z = np.cross(x, y)
    n = np.linalg.norm(z)
    if n < 1e-9:
        y = np.array([0.0, 1.0, 0.0])
        z = np.cross(x, y)
        n = np.linalg.norm(z)
    z = z / n
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    return np.column_stack([x, y, z])


def yaw_from_xy_axis(axis: np.ndarray) -> float:
    v = np.array([axis[0], axis[1], 0.0])
    n = np.linalg.norm(v)
    if n < 1e-9:
        return 0.0
    v /= n
    return float(np.arctan2(v[1], v[0]))


def gripper_frame_of(model: pin.Model, tcp_frame_id: int) -> int:
    name = model.frames[tcp_frame_id].name
    gname = name.replace('linkTCP', 'linkGripper')
    try:
        return model.getFrameId(gname)
    except Exception:
        return tcp_frame_id


def R_tool_down(yaw_rad: float = 0.0) -> np.ndarray:
    """Deprecated alias: gripper +X down."""
    return R_gripper_down(yaw_rad)


def yaw_from_rotation(R: np.ndarray) -> float:
    """Horizontal yaw of frame +Y (used when +X is vertical)."""
    return yaw_from_xy_axis(R[:, 1])


def ik_position(
    model: pin.Model,
    data: pin.Data,
    frame_id: int,
    q0: np.ndarray,
    target: np.ndarray,
    arm_idxs: list[int],
    keep_orientation: bool = False,
    tool_down: bool = True,
    yaw_rad: float | None = None,
    max_iter: int = 80,
    checker: CollisionChecker | None = None,
    n_restarts: int = 4,
) -> tuple[np.ndarray, float, str]:
    """IK to target XYZ. tool_down: gripper +X (夹爪长轴) = world -Z."""

    gid = gripper_frame_of(model, frame_id)

    def gripper_to_tcp_R() -> np.ndarray:
        pin.forwardKinematics(model, data, q0)
        pin.updateFramePlacements(model, data)
        Rg = data.oMf[gid].rotation
        Rt = data.oMf[frame_id].rotation
        return Rg.T @ Rt

    R_gt = gripper_to_tcp_R()

    def desired_R(q_ref: np.ndarray, yaw_override: float | None = None) -> np.ndarray:
        if tool_down:
            if yaw_override is not None:
                yaw = float(yaw_override)
            elif yaw_rad is not None:
                yaw = float(yaw_rad)
            else:
                pin.forwardKinematics(model, data, q_ref)
                pin.updateFramePlacements(model, data)
                yaw = yaw_from_xy_axis(data.oMf[gid].rotation[:, 1])
            return R_gripper_down(yaw) @ R_gt
        if keep_orientation:
            pin.forwardKinematics(model, data, q_ref)
            pin.updateFramePlacements(model, data)
            return data.oMf[frame_id].rotation.copy()
        pin.forwardKinematics(model, data, q_ref)
        pin.updateFramePlacements(model, data)
        return data.oMf[frame_id].rotation.copy()

    def metrics(q: np.ndarray) -> tuple[float, float]:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pos_err = float(np.linalg.norm(data.oMf[frame_id].translation - target))
        gx = data.oMf[gid].rotation[:, 0]
        down_err = float(np.linalg.norm(gx - np.array([0.0, 0.0, -1.0])))
        return pos_err, down_err

    def solve_from(
        q_init: np.ndarray, yaw_override: float | None = None
    ) -> tuple[np.ndarray, float, float]:
        q = q_init.copy()
        # Phase 1: reach XYZ quickly
        for _ in range(max_iter // 2):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            oMf = data.oMf[frame_id]
            err3 = target.reshape(3) - oMf.translation
            if np.linalg.norm(err3) < 4e-3:
                break
            J = pin.computeFrameJacobian(
                model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            Jpos = J[:3, :][:, arm_idxs]
            lam = 1e-3
            dq_arm = Jpos.T @ np.linalg.solve(Jpos @ Jpos.T + lam * np.eye(3), err3)
            dq = np.zeros(model.nv)
            for k, ji in enumerate(arm_idxs):
                dq[ji] = dq_arm[k]
            q = clamp_q(model, pin.integrate(model, q, 0.6 * dq))

        # Phase 2: tip-down
        R_des = desired_R(q, yaw_override)
        for _ in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            oMf = data.oMf[frame_id]
            oMdes = pin.SE3(R_des, target.reshape(3))
            err = pin.log6(oMf.actInv(oMdes)).vector
            if np.linalg.norm(err[:3]) < 3e-3 and np.linalg.norm(err[3:]) < 2e-2:
                break
            J = pin.computeFrameJacobian(
                model, data, q, frame_id, pin.ReferenceFrame.LOCAL
            )
            J_arm = J[:, arm_idxs]
            w = (
                np.array([1.0, 1.0, 1.0, 2.2, 2.2, 2.2])
                if tool_down
                else np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
            )
            Jw = J_arm * w.reshape(6, 1)
            ew = err * w
            lam = 5e-3
            dq_arm = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam * np.eye(6), ew)
            dq = np.zeros(model.nv)
            for k, ji in enumerate(arm_idxs):
                dq[ji] = dq_arm[k]
            q = clamp_q(model, pin.integrate(model, q, 0.45 * dq))

        pos_err, down_err = metrics(q)
        return q, pos_err, down_err

    def accept(pos_err: float, down_err: float, ok: bool) -> bool:
        if not ok:
            return False
        if tool_down:
            return pos_err < 0.035 and down_err < 0.2
        return pos_err < 0.035

    best_q = q0.copy()
    best_err, best_down = metrics(best_q)
    best_status = 'init'
    best_score = 1e9

    # 先无碰撞约束快速解，再校验一次（快一个数量级）
    yaw_list: list[float | None]
    if tool_down and yaw_rad is None:
        yaw_list = [None, 0.0, np.pi / 2, -np.pi / 2]
    else:
        yaw_list = [yaw_rad]

    seeds = [q0.copy()]
    rng = np.random.default_rng(1)
    for _ in range(n_restarts):
        q_seed = q0.copy()
        for ji in arm_idxs:
            lo = model.lowerPositionLimit[ji]
            hi = model.upperPositionLimit[ji]
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo + 1e-6):
                lo, hi = -np.pi, np.pi
            mid = 0.5 * (lo + hi)
            q_seed[ji] = float(rng.uniform(lo, hi) * 0.4 + mid * 0.2 + q0[ji] * 0.4)
        seeds.append(clamp_q(model, q_seed))

    for q_seed in seeds:
        for yaw_try in yaw_list:
            q_try, err_try, down_try = solve_from(q_seed, yaw_try)
            ok = True
            reason = 'ok'
            if checker is not None:
                ok, reason = checker.check(q_try)
            score = err_try + (0.35 * down_try if tool_down else 0.0) + (0.5 if not ok else 0.0)
            if score < best_score:
                best_score = score
                best_q, best_err, best_down = q_try, err_try, down_try
                best_status = (
                    'ok'
                    if accept(err_try, down_try, ok)
                    else (reason if not ok else f'reach_err={err_try:.3f}, down_err={down_try:.3f}')
                )
            if best_status == 'ok':
                return best_q, best_err, 'ok'

    if best_status != 'ok' and checker is not None:
        ok, reason = checker.check(best_q)
        if not ok:
            best_status = reason
    return best_q, best_err, best_status


def interpolate_free(
    q_a: np.ndarray,
    q_b: np.ndarray,
    checker: CollisionChecker | None,
    n: int = 16,
):
    path = []
    for a in np.linspace(0, 1, n):
        qq = (1 - a) * q_a + a * q_b
        if checker is not None:
            ok, _ = checker.check(qq)
            if not ok:
                return None
        path.append(qq)
    return path


def _sample_arm_q(
    model: pin.Model,
    q_template: np.ndarray,
    arm_idxs: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    q = q_template.copy()
    for ji in arm_idxs:
        lo = model.lowerPositionLimit[ji]
        hi = model.upperPositionLimit[ji]
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo + 1e-6):
            lo, hi = -np.pi, np.pi
        q[ji] = float(rng.uniform(lo, hi))
    return clamp_q(model, q)


def _steer_arm(
    model: pin.Model,
    q_from: np.ndarray,
    q_to: np.ndarray,
    arm_idxs: list[int],
    step: float,
) -> np.ndarray:
    dq = np.zeros(model.nv)
    dist = 0.0
    for ji in arm_idxs:
        d = q_to[ji] - q_from[ji]
        dq[ji] = d
        dist += d * d
    dist = np.sqrt(dist)
    if dist < 1e-9:
        return q_from.copy()
    scale = min(1.0, step / dist)
    q = q_from.copy()
    for ji in arm_idxs:
        q[ji] = q_from[ji] + scale * (q_to[ji] - q_from[ji])
    return clamp_q(model, q)


def _nearest(tree: list[np.ndarray], q: np.ndarray, arm_idxs: list[int]) -> int:
    best_i, best_d = 0, 1e18
    for i, node in enumerate(tree):
        d = 0.0
        for ji in arm_idxs:
            dd = node[ji] - q[ji]
            d += dd * dd
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def rrt_connect_arm(
    model: pin.Model,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    arm_idxs: list[int],
    checker: CollisionChecker,
    max_iter: int = 220,
    step: float = 0.35,
    goal_bias: float = 0.35,
    time_budget_s: float = 1.2,
) -> list[np.ndarray] | None:
    """Joint-space RRT-Connect on arm DoFs; other joints fixed."""
    ok_s, _ = checker.check(q_start)
    ok_g, _ = checker.check(q_goal)
    if not ok_s or not ok_g:
        return None
    direct = interpolate_free(q_start, q_goal, checker, n=12)
    if direct is not None:
        return direct

    rng = np.random.default_rng()
    ta: list[np.ndarray] = [q_start.copy()]
    tb: list[np.ndarray] = [q_goal.copy()]
    pa: list[int] = [-1]
    pb: list[int] = [-1]
    t0 = time.time()

    def extend(tree: list[np.ndarray], parents: list[int], q_target: np.ndarray):
        ni = _nearest(tree, q_target, arm_idxs)
        q_new = _steer_arm(model, tree[ni], q_target, arm_idxs, step)
        seg = interpolate_free(tree[ni], q_new, checker, n=3)
        if seg is None:
            return None
        tree.append(q_new)
        parents.append(ni)
        return len(tree) - 1

    def try_join(ia: int, ib: int):
        seg = interpolate_free(ta[ia], tb[ib], checker, n=4)
        if seg is None:
            return None
        path = []
        idx = ia
        while idx >= 0:
            path.append(ta[idx].copy())
            idx = pa[idx]
        path.reverse()
        idx = ib
        rev = []
        while idx >= 0:
            rev.append(tb[idx].copy())
            idx = pb[idx]
        path.extend(rev)
        # 稀疏回放即可，不必再密采样
        dense: list[np.ndarray] = []
        for u in range(len(path) - 1):
            piece = interpolate_free(path[u], path[u + 1], checker, n=3)
            if piece is None:
                return path
            dense.extend(piece[:-1] if u < len(path) - 2 else piece)
        return dense if dense else path

    for _ in range(max_iter):
        if time.time() - t0 > time_budget_s:
            return None
        if rng.random() < goal_bias:
            q_rand = q_goal.copy() if len(ta) <= len(tb) else q_start.copy()
        else:
            q_rand = _sample_arm_q(model, q_start, arm_idxs, rng)
            ok, _ = checker.check(q_rand)
            if not ok:
                continue
        if len(ta) <= len(tb):
            ia = extend(ta, pa, q_rand)
            if ia is None:
                continue
            ib = _nearest(tb, ta[ia], arm_idxs)
            joined = try_join(ia, ib)
            if joined is not None:
                return joined
        else:
            ib = extend(tb, pb, q_rand)
            if ib is None:
                continue
            ia = _nearest(ta, tb[ib], arm_idxs)
            joined = try_join(ia, ib)
            if joined is not None:
                return joined
    return None


def plan_arm_path(
    model: pin.Model,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    arm_idxs: list[int],
    checker: CollisionChecker,
) -> list[np.ndarray] | None:
    # 直线优先（稀疏）
    path = interpolate_free(q_start, q_goal, checker, n=12)
    if path is not None:
        return path

    # 中点抬高关节空间绕行（比 RRT 快很多）
    q_mid = q_start.copy()
    for ji in arm_idxs:
        q_mid[ji] = 0.5 * (q_start[ji] + q_goal[ji])
    # 朝中立附近抬一下，减少扫到桌面
    q_n = pin.neutral(model)
    for ji in arm_idxs[:3]:
        q_mid[ji] = 0.55 * q_mid[ji] + 0.45 * q_n[ji]
    q_mid = clamp_q(model, q_mid)
    if checker.check(q_mid)[0]:
        p1 = interpolate_free(q_start, q_mid, checker, n=8)
        p2 = interpolate_free(q_mid, q_goal, checker, n=8)
        if p1 is not None and p2 is not None:
            return p1 + p2[1:]

    # 短时 RRT：规划期关掉自碰，只避点云
    print('直线碰撞，快速绕障…')
    prev_fast = checker.fast_mode
    checker.fast_mode = True
    try:
        path = rrt_connect_arm(
            model, q_start, q_goal, arm_idxs, checker, time_budget_s=1.0
        )
    finally:
        checker.fast_mode = prev_fast
    if path is not None:
        print(f'绕障路径 {len(path)} 节点')
        return path
    print('绕障失败')
    return None


def ik_with_detours(
    model: pin.Model,
    data: pin.Data,
    frame_id: int,
    q0: np.ndarray,
    target: np.ndarray,
    arm_idxs: list[int],
    checker: CollisionChecker | None,
    tool_down: bool,
) -> tuple[np.ndarray, float, str]:
    """IK to target; if blocked, try elevated / side approaches then final."""
    q_goal, err, st = ik_position(
        model, data, frame_id, q0, target, arm_idxs, checker=checker, tool_down=tool_down
    )
    if st == 'ok':
        return q_goal, err, st

    # Cartesian detours above / aside the target
    offsets = [
        np.array([0.0, 0.0, 0.12]),
        np.array([0.0, 0.0, 0.22]),
        np.array([0.08, 0.0, 0.12]),
        np.array([-0.08, 0.0, 0.12]),
    ]
    best = (q_goal, err, st)
    for off in offsets:
        q_via, err_v, st_v = ik_position(
            model,
            data,
            frame_id,
            q0,
            target + off,
            arm_idxs,
            checker=checker,
            tool_down=tool_down,
        )
        if st_v != 'ok':
            continue
        q_fin, err_f, st_f = ik_position(
            model,
            data,
            frame_id,
            q_via,
            target,
            arm_idxs,
            checker=checker,
            tool_down=tool_down,
        )
        if st_f == 'ok':
            return q_fin, err_f, 'ok_via_detour'
        if err_f < best[1]:
            best = (q_fin, err_f, st_f)
    return best


def set_marker(viewer, name: str, xyz: np.ndarray, color: int, radius: float = 0.028):
    viewer[name].set_object(
        mcg.Sphere(radius),
        mcg.MeshPhongMaterial(color=color, opacity=0.92, reflectivity=0.3),
    )
    T = np.eye(4)
    T[:3, 3] = xyz
    viewer[name].set_transform(T)


def set_waypoint_path(viewer, points: list[np.ndarray]):
    """路径线已关闭（演示页不画轨迹）。"""
    try:
        viewer['pick'].delete()
    except Exception:
        pass


def clear_pick_viz(viewer) -> None:
    try:
        viewer['pick'].delete()
    except Exception:
        pass


def animate_gripper_pinch(
    viewer,
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    tcp_frame_id: int,
    arm: str,
    step_sleep: float,
    q_traj_acc: list | None = None,
    grasp_gaps: list | None = None,
) -> None:
    """URDF 夹爪无开合关节：Meshcat 示意合拢→张开；并写入轨迹供 Foxglove。"""
    root = f'demo_grasp/{arm}'
    mat = mcg.MeshPhongMaterial(color=0xFBBF24, opacity=0.95, reflectivity=0.25)
    for side in ('a', 'b'):
        viewer[f'{root}/{side}'].set_object(mcg.Box([0.05, 0.01, 0.018]), mat)

    open_half = 0.030
    close_half = 0.003

    def place(gap_half: float) -> None:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        oMf = data.oMf[tcp_frame_id]
        R = oMf.rotation
        p = oMf.translation
        base = p - 0.015 * R[:, 0]
        y = R[:, 1]
        for sign, name in ((1.0, 'a'), (-1.0, 'b')):
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = base + sign * gap_half * y
            viewer[f'{root}/{name}'].set_transform(T)
        if q_traj_acc is not None:
            q_traj_acc.append(np.asarray(q, float).tolist())
            if grasp_gaps is not None:
                grasp_gaps.append(float(gap_half))

    place(open_half)
    n_close = 16
    for k in range(1, n_close + 1):
        g = open_half + (close_half - open_half) * (k / n_close)
        place(g)
        time.sleep(max(0.02, step_sleep * 1.15))
    # 合并停顿：重复闭合位姿若干帧，导出让 Foxglove 看得见停顿
    for _ in range(8):
        place(close_half)
        time.sleep(0.05)
    n_open = 14
    for k in range(1, n_open + 1):
        g = close_half + (open_half - close_half) * (k / n_open)
        place(g)
        time.sleep(max(0.02, step_sleep * 1.15))
    try:
        viewer[root].delete()
    except Exception:
        pass


def history_dir(arm: str) -> Path:
    d = HISTORY_ROOT / ('L' if arm.upper() == 'L' else 'R')
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_history(arm: str) -> list[dict]:
    items = []
    for p in sorted(history_dir(arm).glob('*.json'), reverse=True):
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        items.append(
            {
                'id': p.stem,
                'file': p.name,
                'arm': obj.get('arm', arm),
                'created': obj.get('created', ''),
                'n_points': len(obj.get('points') or []),
                'n_q': len(obj.get('q_traj') or []),
                'note': obj.get('note') or '',
                'hover_m': (obj.get('opts') or {}).get('hover_m'),
                'speed': (obj.get('opts') or {}).get('speed'),
            }
        )
    return items


def save_history_record(arm: str, record: dict) -> Path:
    ts = time.strftime('%Y%m%d_%H%M%S')
    n = len(record.get('points') or [])
    path = history_dir(arm) / f'{ts}_{arm}_{n}pts.json'
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def load_history_record(arm: str, hist_id: str) -> dict | None:
    path = history_dir(arm) / f'{hist_id}.json'
    if not path.is_file():
        # allow bare filename
        cand = history_dir(arm) / hist_id
        path = cand if cand.is_file() else path
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def style_meshcat_scene(viz: MeshcatVisualizer, viewer) -> None:
    """Dark slate Meshcat scene (replaces default blue)."""
    viewer['/Background'].set_property('top_color', [0.07, 0.09, 0.11])
    viewer['/Background'].set_property('bottom_color', [0.16, 0.18, 0.21])
    viewer['/Grid'].set_property('visible', True)
    try:
        viewer['/Grid'].set_property('color', [0.38, 0.42, 0.46])
    except Exception:
        pass
    viewer['/Axes'].set_property('visible', False)
    for path in (
        '/Lights/FillLight2',
        '/Lights/SpotLight',
        '/Lights/DirectionalLight',
        '/Lights/AmbientLight',
    ):
        try:
            viewer[path].set_property('visible', True)
        except Exception:
            pass
    try:
        viewer['/Lights/AmbientLight/<object>'].set_property('intensity', 0.7)
    except Exception:
        pass
    try:
        viz.setCameraTarget([0.28, 0.0, 0.52])
        viz.setCameraPosition([1.55, -1.45, 1.05])
        viz.setCameraZoom(0.9)
    except Exception:
        pass


UI_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Rimbot Demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,500;9..40,650&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
:root{
  --panel:#12171e; --panel2:#181e27; --line:#2b3442; --text:#eef3f9; --muted:#9aa8bc;
  --accent:#5eead4; --warn:#fbbf24; --danger:#fb7185; --ok:#34d399;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{
  color:var(--text); font-family:"DM Sans",system-ui,sans-serif;
  background:#0b0e13; overflow:hidden;
}
.app{
  display:grid; grid-template-columns:minmax(300px,360px) minmax(0,1fr);
  grid-template-rows:100vh; height:100vh; width:100vw;
}
@media(max-width:980px){
  body{overflow:auto}
  .app{grid-template-columns:1fr; grid-template-rows:auto minmax(55vh,70vh); height:auto; min-height:100vh}
}
.side{
  background:linear-gradient(180deg,#151b24,#10151c);
  border-right:1px solid var(--line);
  padding:16px 14px 18px; overflow:auto; min-height:0;
}
.brand{font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--accent); font-weight:650}
h1{margin:6px 0 4px; font-size:22px; font-weight:650; letter-spacing:-.02em}
.sub{margin:0 0 12px; color:var(--muted); font-size:13px; line-height:1.45}
.stage{position:relative; background:#0a0d12; min-width:0; min-height:0; overflow:hidden}
#meshcat-pane{position:absolute; inset:0; width:100%; height:100%; overflow:hidden}
#meshHint{
  position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  color:var(--muted); font-size:14px; pointer-events:none; z-index:1;
}
.card{
  background:var(--panel2); border:1px solid var(--line); border-radius:14px;
  padding:12px; margin-bottom:12px;
}
.card h2{margin:0 0 8px; font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted)}
.row{display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:6px 0}
label{display:inline-flex; gap:6px; align-items:center; color:var(--muted); font-size:12.5px}
.checks label{padding:5px 9px; border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.02)}
select,input[type=number]{
  background:#0d1218; border:1px solid var(--line); color:var(--text);
  border-radius:9px; padding:7px 8px; font-family:"JetBrains Mono",monospace; font-size:12px;
}
input[type=number]{width:84px}
button{
  font:inherit; border:1px solid var(--line); background:rgba(255,255,255,.03);
  color:var(--text); border-radius:10px; padding:8px 11px; cursor:pointer;
}
button:hover{border-color:var(--accent)}
.big{background:linear-gradient(135deg,#2dd4bf,#0d9488); color:#042f2e; border:0; font-weight:650}
.box{position:relative; border:1px solid var(--line); border-radius:12px; overflow:hidden; background:#0a0e14}
canvas{display:block; width:100%; height:auto; cursor:crosshair}
#status{
  margin-top:8px; padding:9px 10px; border-radius:10px; font-size:12.5px; color:var(--muted);
  background:rgba(0,0,0,.25); border:1px solid var(--line); min-height:2.4em;
}
#status.err{color:var(--danger)} #status.ok{color:var(--ok)}
table{width:100%; border-collapse:collapse; font-family:"JetBrains Mono",monospace; font-size:11.5px}
th,td{padding:4px 4px; border-bottom:1px solid var(--line); text-align:left; vertical-align:middle}
th{color:var(--muted); font-size:10px; letter-spacing:.06em; text-transform:uppercase}
td input.wpt{
  width:72px; padding:4px 5px; background:#0d1218; border:1px solid var(--line);
  color:var(--text); border-radius:7px; font:inherit;
}
#cloudMsg{color:var(--muted); font-size:12px; margin-top:6px}
input[type=range]{width:120px; accent-color:var(--accent)}
</style></head><body>
<div class="app">
  <aside class="side">
    <div class="brand">Rimbot Demo</div>
    <h1>途径点控制</h1>
    <p class="sub">勾选「旋转+降」的点：到达后原地打转 → 下降 → 夹合一下再张开 → 回到该点 → 再走下一个点。可勾选结束后复原位。</p>

    <div class="card">
      <h2>执行</h2>
      <div class="row">
        <label>手臂
          <select id="arm"><option value="L">左臂 L</option><option value="R">右臂 R</option></select>
        </label>
      </div>
      <div class="row checks">
        <label><input type="checkbox" id="snapCloud"> 吸附点云Z</label>
        <label><input type="checkbox" id="avoidCloud" checked> 避障点云</label>
        <label><input type="checkbox" id="avoidSelf" checked> 避自碰</label>
        <label><input type="checkbox" id="toolDown" checked> 末端向下</label>
        <label><input type="checkbox" id="returnHome" checked> 结束后复原位</label>
      </div>
      <div class="row">
        <label>上方悬停(m) <input type="number" step="0.05" id="hoverZ" value="0.20"></label>
        <label>速度 <input type="range" id="speed" min="0.1" max="2" step="0.05" value="0.35" oninput="document.getElementById('speedVal').textContent=this.value+'×'"></label>
        <span id="speedVal" style="color:var(--muted);font-family:JetBrains Mono,monospace;font-size:12px">0.35×</span>
        <label>到点停顿(s) <input type="number" step="0.5" id="dwellS" value="2" min="0" style="width:64px"></label>
      </div>
      <div class="row">
        <label>选中点原地打转(°) <input type="number" step="5" id="twistDeg" value="45" style="width:72px"></label>
        <label>再下降(m) <input type="number" step="0.01" id="dropZ" value="0.10" style="width:72px"></label>
        <label>点云膨胀 <input type="number" step="0.005" id="infCloud" value="0.03"></label>
        <label>自碰膨胀 <input type="number" step="0.005" id="infSelf" value="0.02"></label>
      </div>
      <div class="row">
        <label>x <input type="number" step="0.01" id="wx" value="0.35"></label>
        <label>y <input type="number" step="0.01" id="wy" value="0.15"></label>
        <label>z <input type="number" step="0.01" id="wz" value="0.85"></label>
      </div>
      <div class="row">
        <button onclick="addManual()">添加点</button>
        <button onclick="delLast()">删末点</button>
        <button onclick="clearPts()">清空</button>
        <button onclick="goHome()">复原位</button>
        <button class="big" onclick="go()">执行</button>
      </div>
      <div id="status">就绪</div>
    </div>

    <div class="card">
      <h2>俯视图点选</h2>
      <div class="box"><canvas id="cv" width="640" height="420"></canvas></div>
      <div id="cloudMsg"></div>
      <table>
        <thead><tr><th>#</th><th>x</th><th>y</th><th>z</th><th>旋转+降</th><th></th></tr></thead>
        <tbody id="wptBody"></tbody>
      </table>
    </div>

    <div class="card">
      <h2>轨迹历史（按臂）</h2>
      <div class="row">
        <button onclick="refreshHistory()">刷新</button>
        <button onclick="saveHistoryNow()">保存当前点</button>
      </div>
      <div id="histList" style="margin-top:8px;max-height:220px;overflow:auto;font-size:12px;color:var(--muted)"></div>
      <div class="row" style="margin-top:8px">
        <button onclick="exportFoxglove()">导出最近一次 → Foxglove MCAP</button>
      </div>
      <div id="foxMsg" style="margin-top:6px;font-size:12px;color:var(--muted)"></div>
    </div>
  </aside>

  <main class="stage">
    <div id="meshHint">正在加载 3D…</div>
    <div id="meshcat-pane"></div>
  </main>
</div>
<script src="/mc/main.min.js"></script>
<script>
(function(){
  const hint = document.getElementById('meshHint');
  try {
    const pane = document.getElementById('meshcat-pane');
    const viewer3d = new MeshCat.Viewer(pane);
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    viewer3d.connect(proto + '://' + location.host + '/mc/ws');
    hint.style.display = 'none';
  } catch (e) {
    hint.textContent = '3D 加载失败: ' + e;
  }
})();
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let cloud = null;
let picks = [];

function setStatus(t, kind){
  const el=document.getElementById('status');
  el.textContent=t;
  el.className = kind==='err' ? 'err' : (kind==='ok' ? 'ok' : '');
}
function renderTable(){
  const tb=document.getElementById('wptBody');
  tb.innerHTML='';
  picks.forEach((p,i)=>{
    if(p.do_twist===undefined) p.do_twist=false;
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${i+1}</td>
      <td><input class="wpt" type="number" step="0.01" value="${(+p.x).toFixed(3)}" onchange="editPt(${i},'x',this.value)"></td>
      <td><input class="wpt" type="number" step="0.01" value="${(+p.y).toFixed(3)}" onchange="editPt(${i},'y',this.value)"></td>
      <td><input class="wpt" type="number" step="0.01" value="${(+p.z).toFixed(3)}" onchange="editPt(${i},'z',this.value)"></td>
      <td><input type="checkbox" ${p.do_twist?'checked':''} onchange="setTwist(${i},this.checked)" title="到达此点后做末端旋转+下降"></td>
      <td><button onclick="delAt(${i})">删</button></td>`;
    tb.appendChild(tr);
  });
}
function editPt(i, axis, val){
  const v=Number(val);
  if(!Number.isFinite(v) || !picks[i]) return;
  picks[i][axis]=v;
  draw(); syncMarkers();
  setStatus(`已改 #${i+1} ${axis}=${v.toFixed(3)}`);
}
function setTwist(i, on){
  if(!picks[i]) return;
  picks[i].do_twist=!!on;
  setStatus(on?`#${i+1} 将做末端旋转+下降`:`#${i+1} 取消旋转动作`);
}
function syncMarkers(){
  // 3D 不再画途经点/轨迹线，只同步清空
  fetch('/api/clear_markers',{method:'POST'});
}
async function refreshHistory(){
  const arm=document.getElementById('arm').value;
  const raw=await (await fetch('/api/history?arm='+arm)).json();
  const items=Array.isArray(raw)?raw:(raw.items||[]);
  const box=document.getElementById('histList');
  if(!items.length){ box.textContent='（暂无历史）'; return; }
  box.innerHTML='';
  items.forEach(it=>{
    const row=document.createElement('div');
    row.style.cssText='display:flex;gap:6px;align-items:center;flex-wrap:wrap;padding:6px 0;border-bottom:1px solid var(--line)';
    row.innerHTML=`<span style="flex:1;min-width:140px;font-family:JetBrains Mono,monospace">${it.created||it.id}<br/>${it.n_points}点 / ${it.n_q}关节帧</span>
      <button onclick="loadHist('${it.id}')">加载点</button>
      <button class="big" onclick="replayHist('${it.id}')">回放</button>
      <button onclick="delHist('${it.id}')">删</button>`;
    box.appendChild(row);
  });
}
async function saveHistoryNow(){
  if(picks.length<1){ setStatus('先加点再保存','err'); return; }
  const out=await (await fetch('/api/history/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      arm:document.getElementById('arm').value, points:picks,
      opts:{
        hover_m:+document.getElementById('hoverZ').value,
        speed:+document.getElementById('speed').value,
        dwell_s:+document.getElementById('dwellS').value,
        avoid_cloud:document.getElementById('avoidCloud').checked,
        avoid_self:document.getElementById('avoidSelf').checked,
        tool_down:document.getElementById('toolDown').checked,
        end_twist_deg:+document.getElementById('twistDeg').value,
        end_drop_m:+document.getElementById('dropZ').value,
        return_home:document.getElementById('returnHome').checked,
      },
      note:'手动保存途径点',
    })})).json();
  setStatus(out.message||JSON.stringify(out), out.ok?'ok':'err');
  refreshHistory();
}
async function loadHist(id){
  const arm=document.getElementById('arm').value;
  const out=await (await fetch('/api/history/load',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({arm,id})})).json();
  if(!out.ok){ setStatus(out.message||'加载失败','err'); return; }
  picks=(out.points||[]).map(p=>({x:+p.x,y:+p.y,z:+p.z, do_twist:!!p.do_twist}));
  if(out.opts){
    if(out.opts.hover_m!=null) document.getElementById('hoverZ').value=out.opts.hover_m;
    if(out.opts.speed!=null){
      document.getElementById('speed').value=out.opts.speed;
      document.getElementById('speedVal').textContent=out.opts.speed+'×';
    }
    if(out.opts.dwell_s!=null) document.getElementById('dwellS').value=out.opts.dwell_s;
    if(out.opts.return_home!=null) document.getElementById('returnHome').checked=!!out.opts.return_home;
    if(out.opts.end_twist_deg!=null) document.getElementById('twistDeg').value=out.opts.end_twist_deg;
    if(out.opts.end_drop_m!=null) document.getElementById('dropZ').value=out.opts.end_drop_m;
  }
  renderTable(); draw(); syncMarkers();
  setStatus(`已加载历史 ${id}（${picks.length} 点）`,'ok');
}
async function replayHist(id){
  setStatus('回放历史…');
  const out=await (await fetch('/api/history/replay',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      arm:document.getElementById('arm').value, id,
      speed:+document.getElementById('speed').value,
      dwell_s:+document.getElementById('dwellS').value,
    })})).json();
  if(!out.ok){ setStatus(out.message||'回放失败','err'); return; }
  setStatus(out.message||'回放中…');
  for(;;){
    await new Promise(r=>setTimeout(r,300));
    const st=await (await fetch('/api/status')).json();
    setStatus(st.message||(st.busy?'回放中…':'回放完成'), st.busy?'':'ok');
    if(!st.busy) break;
  }
}
async function delHist(id){
  const arm=document.getElementById('arm').value;
  const out=await (await fetch('/api/history/delete',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({arm,id})})).json();
  setStatus(out.message||'', out.ok?'ok':'err');
  refreshHistory();
}
async function exportFoxglove(){
  const el=document.getElementById('foxMsg');
  el.textContent='正在导出 MCAP…';
  const out=await (await fetch('/api/export_foxglove',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({arm:document.getElementById('arm').value})})).json();
  el.textContent = out.message || JSON.stringify(out);
  el.style.color = out.ok ? 'var(--ok)' : 'var(--danger)';
}
document.getElementById('arm').addEventListener('change', refreshHistory);
function worldToPix(x,y){
  const b=cloud||{xmin:-0.5,xmax:0.8,ymin:-0.5,ymax:0.5};
  const s=Math.min((cv.width-48)/(b.xmax-b.xmin||1),(cv.height-48)/(b.ymax-b.ymin||1));
  const cx=(b.xmin+b.xmax)/2, cy=(b.ymin+b.ymax)/2;
  return [cv.width/2+(x-cx)*s, cv.height/2-(y-cy)*s];
}
function pixToWorld(px,py){
  const b=cloud||{xmin:-0.5,xmax:0.8,ymin:-0.5,ymax:0.5};
  const s=Math.min((cv.width-48)/(b.xmax-b.xmin||1),(cv.height-48)/(b.ymax-b.ymin||1));
  const cx=(b.xmin+b.xmax)/2, cy=(b.ymin+b.ymax)/2;
  return [cx+(px-cv.width/2)/s, cy-(py-cv.height/2)/s];
}
function draw(){
  const g=ctx.createLinearGradient(0,0,0,cv.height);
  g.addColorStop(0,'#121820'); g.addColorStop(1,'#0a0e14');
  ctx.fillStyle=g; ctx.fillRect(0,0,cv.width,cv.height);
  ctx.strokeStyle='rgba(148,163,184,.07)';
  for(let x=32;x<cv.width;x+=32){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,cv.height);ctx.stroke();}
  for(let y=32;y<cv.height;y+=32){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(cv.width,y);ctx.stroke();}
  if(cloud && cloud.xs && cloud.xs.length){
    for(let i=0;i<cloud.xs.length;i++){
      const [px,py]=worldToPix(cloud.xs[i], cloud.ys[i]);
      const t=(cloud.zs[i]-cloud.zmin)/((cloud.zmax-cloud.zmin)||1);
      ctx.fillStyle=`rgba(${50+110*t|0},${100+70*(1-t)|0},${140+80*t|0},0.9)`;
      ctx.fillRect(px,py,2,2);
    }
  } else {
    ctx.fillStyle='rgba(154,168,188,.9)'; ctx.font='14px DM Sans,sans-serif';
    ctx.fillText('点击任意位置加点', 28, 40);
  }
  if(picks.length>1){
    ctx.strokeStyle='rgba(94,234,212,.55)'; ctx.lineWidth=2; ctx.beginPath();
    picks.forEach((p,i)=>{ const [px,py]=worldToPix(p.x,p.y); if(i===0)ctx.moveTo(px,py); else ctx.lineTo(px,py); });
    ctx.stroke();
  }
  const colors=['#fb7185','#34d399','#60a5fa','#fbbf24','#e879f9','#5eead4','#f8fafc'];
  picks.forEach((p,i)=>{
    const [px,py]=worldToPix(p.x,p.y); const c=colors[i%colors.length];
    ctx.beginPath(); ctx.fillStyle=c+'33'; ctx.arc(px,py,14,0,6.28); ctx.fill();
    ctx.strokeStyle=c; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(px,py,8,0,6.28); ctx.stroke();
    ctx.fillStyle=c; ctx.font='600 12px JetBrains Mono,monospace'; ctx.fillText(String(i+1), px+10, py-6);
  });
}
async function loadCloud(){
  try{
    cloud=await (await fetch('/api/cloud')).json();
    const n=(cloud.xs&&cloud.xs.length)||0;
    document.getElementById('cloudMsg').textContent = n ? `点云参考 ${n.toLocaleString()} pts · 任意位置可点` : '无点云 · 仍可自由加点';
  }catch(e){ document.getElementById('cloudMsg').textContent='点云未加载'; }
  draw();
}
function addPoint(p){
  picks.push({x:+p.x,y:+p.y,z:+p.z, do_twist:!!p.do_twist});
  renderTable(); draw(); syncMarkers();
  setStatus(`已添加 #${picks.length}`);
}
function addManual(){
  addPoint({
    x:+document.getElementById('wx').value,
    y:+document.getElementById('wy').value,
    z:+document.getElementById('wz').value,
  });
}
function delLast(){ if(picks.length){ picks.pop(); renderTable(); draw(); syncMarkers(); } }
function delAt(i){ picks.splice(i,1); renderTable(); draw(); syncMarkers(); }
async function clearPts(){ picks=[]; renderTable(); draw(); await fetch('/api/clear_markers',{method:'POST'}); setStatus('已清空'); }
cv.addEventListener('click', async (ev)=>{
  const r=cv.getBoundingClientRect();
  const px=(ev.clientX-r.left)*(cv.width/r.width);
  const py=(ev.clientY-r.top)*(cv.height/r.height);
  const [x,y]=pixToWorld(px,py);
  if(document.getElementById('snapCloud').checked){
    const s=await (await fetch('/api/snap',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x,y})})).json();
    if(s.error){ setStatus(s.error,'err'); return; }
    addPoint(s);
  } else {
    addPoint({x,y,z:+document.getElementById('wz').value});
  }
});
async function waitBusy(startMsg){
  setStatus(startMsg||'运动中…');
  for(;;){
    await new Promise(r=>setTimeout(r,350));
    const st=await (await fetch('/api/status')).json();
    setStatus(st.message||(st.busy?'运动中…':'完成'), st.busy?'':'ok');
    if(!st.busy) break;
  }
}
async function goHome(){
  setStatus('复原位…');
  const out=await (await fetch('/api/home',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      arm:document.getElementById('arm').value,
      avoid_cloud:document.getElementById('avoidCloud').checked,
      avoid_self:document.getElementById('avoidSelf').checked,
      inflate_cloud_m:+document.getElementById('infCloud').value,
      inflate_self_m:+document.getElementById('infSelf').value,
      speed:+document.getElementById('speed').value,
    })})).json();
  if(!out.ok){ setStatus(out.message||JSON.stringify(out),'err'); return; }
  await waitBusy(out.message||'复原位中…');
}
async function go(){
  if(picks.length<1){ setStatus('先添加途径点','err'); return; }
  setStatus(`运动中（${picks.length} 点）…`);
  const out=await (await fetch('/api/goto',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      arm:document.getElementById('arm').value, points:picks,
      avoid_cloud:document.getElementById('avoidCloud').checked,
      avoid_self:document.getElementById('avoidSelf').checked,
      tool_down:document.getElementById('toolDown').checked,
      inflate_cloud_m:+document.getElementById('infCloud').value,
      inflate_self_m:+document.getElementById('infSelf').value,
      hover_m:+document.getElementById('hoverZ').value,
      speed:+document.getElementById('speed').value,
      dwell_s:+document.getElementById('dwellS').value,
      end_twist_deg:+document.getElementById('twistDeg').value,
      end_drop_m:+document.getElementById('dropZ').value,
      return_home:document.getElementById('returnHome').checked,
    })})).json();
  if(!out.ok){ setStatus(out.message||JSON.stringify(out),'err'); return; }
  await waitBusy(out.message||'运动中…');
}
renderTable(); draw(); loadCloud(); refreshHistory();
</script></body></html>
"""


def _waypoint_color(i: int) -> int:
    palette = (0xFB7185, 0x34D399, 0x60A5FA, 0xFBBF24, 0xE879F9, 0x5EEAD4, 0xF8FAFC, 0x94A3B8)
    return palette[i % len(palette)]
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    ap.add_argument('--ply', type=Path, default=DEFAULT_PLY)
    ap.add_argument('--max-points', type=int, default=200000)
    ap.add_argument('--point-size', type=float, default=0.009)
    ap.add_argument('--cloud-x-offset-m', type=float, default=0.1)
    ap.add_argument('--cloud-y-offset-m', type=float, default=0.0)
    ap.add_argument('--cloud-z-offset-m', type=float, default=0.75)
    ap.add_argument('--ui-port', type=int, default=8765)
    ap.add_argument('--host', default='0.0.0.0')
    args = ap.parse_args()

    if not args.urdf.is_file():
        raise SystemExit(f'missing urdf: {args.urdf}')

    tmp = Path('/tmp') / f'{args.urdf.stem}_abs.urdf'
    urdf_abs = prepare_urdf_abs(args.urdf, tmp)

    model, collision_model, visual_model = pin.buildModelsFromUrdf(str(urdf_abs))
    collision_for_check = collision_model
    try:
        build_self_collision_model(model, collision_for_check)
        print(f'自碰检测对: {len(collision_for_check.collisionPairs)}')
    except Exception as e:
        print(f'警告: 自碰对构建失败: {e}')
        collision_for_check = None

    data = model.createData()
    meta = joint_meta(model)
    q = pin.neutral(model)

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=False)
    viz.loadViewerModel()
    viz.display(q)
    viewer = viz.viewer
    style_meshcat_scene(viz, viewer)

    cloud_xyz = None
    if args.ply.is_file():
        print(f'加载点云 {args.ply} …')
        xyz, rgb = load_ply_xyzrgb(args.ply, max_points=args.max_points)
        dx, dy, dz = (
            float(args.cloud_x_offset_m),
            float(args.cloud_y_offset_m),
            float(args.cloud_z_offset_m),
        )
        xyz = xyz.copy()
        xyz[0] += dx
        xyz[1] += dy
        xyz[2] += dz
        print(f'点云偏移 Δ=({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) m')
        viewer['map'].set_object(mcg.PointCloud(xyz, rgb, size=args.point_size))
        cloud_xyz = xyz
        print(f'点云已加入 Meshcat: {xyz.shape[1]} pts')
    else:
        print(f'警告: 无点云 {args.ply}，俯视图选点不可用')

    # downsample for canvas
    cloud_ui = {'xs': [], 'ys': [], 'zs': [], 'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1, 'zmin': 0, 'zmax': 1}
    if cloud_xyz is not None:
        n = cloud_xyz.shape[1]
        step = max(1, n // 8000)
        sub = cloud_xyz[:, ::step]
        cloud_ui = {
            'xs': sub[0].tolist(),
            'ys': sub[1].tolist(),
            'zs': sub[2].tolist(),
            'xmin': float(sub[0].min()),
            'xmax': float(sub[0].max()),
            'ymin': float(sub[1].min()),
            'ymax': float(sub[1].max()),
            'zmin': float(sub[2].min()),
            'zmax': float(sub[2].max()),
        }

    state = {
        'q': q,
        'lock': threading.Lock(),
        'busy': False,
        'last_msg': '就绪',
        'last_history': None,
        'last_joints_json': None,
    }
    ik_data = model.createData()
    checker = CollisionChecker(
        model,
        ik_data,
        collision_for_check,
        cloud_xyz,
        inflate_self_m=0.02,
        inflate_cloud_m=0.03,
        enable_self=collision_for_check is not None,
        enable_cloud=cloud_xyz is not None,
    )

    def apply_q(new_q: np.ndarray) -> None:
        with state['lock']:
            state['q'] = new_q.copy()
            viz.display(state['q'])

    def snap_xy(x: float, y: float, z: float | None = None) -> dict:
        if cloud_xyz is None:
            return {'error': '无点云'}
        d2 = (cloud_xyz[0] - x) ** 2 + (cloud_xyz[1] - y) ** 2
        if z is not None:
            d2 = d2 + (cloud_xyz[2] - float(z)) ** 2
        i = int(np.argmin(d2))
        return {
            'x': float(cloud_xyz[0, i]),
            'y': float(cloud_xyz[1, i]),
            'z': float(cloud_xyz[2, i]),
        }

    def animate_goto(arm: str, points: list[dict], opts: dict | None = None) -> dict:
        opts = opts or {}
        if state['busy']:
            return {'ok': False, 'message': '正在运动，请稍候'}
        if not points:
            return {'ok': False, 'message': '没有途径点'}
        checker.set_params(
            inflate_self_m=float(opts.get('inflate_self_m', 0.02)),
            inflate_cloud_m=float(opts.get('inflate_cloud_m', 0.03)),
            enable_self=bool(opts.get('avoid_self', True)),
            enable_cloud=bool(opts.get('avoid_cloud', True)),
            active_arm=arm,
        )
        frame_id = resolve_frame(model, arm)
        arm_idxs = arm_q_indices(model, arm)
        if len(arm_idxs) < 7:
            return {'ok': False, 'message': f'关节索引异常: {arm_idxs}'}

        hover_m = float(opts.get('hover_m', 0.20))
        speed = float(opts.get('speed', 0.35))
        speed = max(0.05, min(5.0, speed))
        dwell_s = float(opts.get('dwell_s', 2.0))
        dwell_s = max(0.0, min(30.0, dwell_s))
        # 更慢的默认插值节拍
        step_sleep = 0.035 / speed
        hold_sleep = dwell_s
        surface_pts = [
            np.array([float(p['x']), float(p['y']), float(p['z'])], dtype=float)
            for p in points
        ]
        targets = [p + np.array([0.0, 0.0, hover_m]) for p in surface_pts]
        clear_pick_viz(viewer)

        tool_down = bool(opts.get('tool_down', True))
        return_home = bool(opts.get('return_home', True))
        state['busy'] = True
        save_opts = {
            'hover_m': hover_m,
            'speed': speed,
            'dwell_s': dwell_s,
            'avoid_cloud': bool(opts.get('avoid_cloud', True)),
            'avoid_self': bool(opts.get('avoid_self', True)),
            'tool_down': tool_down,
            'inflate_cloud_m': float(opts.get('inflate_cloud_m', 0.03)),
            'inflate_self_m': float(opts.get('inflate_self_m', 0.02)),
            'end_twist_deg': float(opts.get('end_twist_deg', 45.0)),
            'end_drop_m': float(opts.get('end_drop_m', 0.10)),
            'return_home': return_home,
        }

        do_twist_flags = [bool(p.get('do_twist')) for p in points]
        grasp_gaps: list[float | None] = [None]  # 与 q_traj 对齐；夹合时写半开合距

        def _push_q(q_traj_acc: list, qq, gap: float | None = None) -> None:
            q_traj_acc.append(np.asarray(qq, float).tolist())
            grasp_gaps.append(gap)

        def play_q_path(path_qs, q_traj_acc):
            for qq in path_qs:
                apply_q(qq)
                _push_q(q_traj_acc, qq)
                time.sleep(step_sleep)

        def arm_home_q(q_now: np.ndarray) -> np.ndarray:
            q_h = q_now.copy()
            q_n = pin.neutral(model)
            for ji in arm_idxs:
                q_h[ji] = float(q_n[ji])
            return clamp_q(model, q_h)

        def go_home_arm(q_now: np.ndarray, q_traj_acc: list, tag: str = '') -> np.ndarray:
            q_goal = arm_home_q(q_now)
            state['last_msg'] = f'{tag}复原位…'
            path = plan_arm_path(model, q_now, q_goal, arm_idxs, checker)
            if path is None:
                prev_cloud = checker.enable_cloud
                checker.enable_cloud = False
                try:
                    path = plan_arm_path(model, q_now, q_goal, arm_idxs, checker)
                finally:
                    checker.enable_cloud = prev_cloud
            if path is None:
                path = interpolate_free(q_now, q_goal, None, n=24)
            if path is None:
                path = [
                    q_now + (q_goal - q_now) * (k / 24.0) for k in range(1, 25)
                ]
                path[-1] = q_goal
            play_q_path(path, q_traj_acc)
            apply_q(q_goal)
            _push_q(q_traj_acc, q_goal)
            print(f'{tag}已复原位')
            return q_goal.copy()

        def do_end_twist_and_drop(
            q_now: np.ndarray,
            q_traj_acc: list,
            q_return: np.ndarray,
            tag: str = '',
        ) -> np.ndarray:
            """原地打转 → Z 下降 → 夹合再张开 → 关节空间回该点位姿。"""
            twist_deg = float(opts.get('end_twist_deg', 45.0))
            drop_m = abs(float(opts.get('end_drop_m', 0.10)))
            gid = gripper_frame_of(model, frame_id)
            q_hold = np.asarray(q_return, float).copy()

            pin.forwardKinematics(model, ik_data, q_now)
            pin.updateFramePlacements(model, ik_data)
            p_hold = ik_data.oMf[frame_id].translation.copy()
            yaw0 = yaw_from_xy_axis(ik_data.oMf[gid].rotation[:, 1])
            yaw1 = yaw0 + np.deg2rad(twist_deg)

            state['last_msg'] = f'{tag}末端原地打转 {twist_deg:.0f}°…'
            n_spin = max(16, int(abs(twist_deg) / 4.0) + 8)
            q_spin = q_now.copy()
            st_s, err_s = 'ok', 0.0
            for k in range(1, n_spin + 1):
                yaw = yaw0 + (yaw1 - yaw0) * (k / n_spin)
                q_spin, err_s, st_s = ik_position(
                    model,
                    ik_data,
                    frame_id,
                    q_spin,
                    p_hold,
                    arm_idxs,
                    checker=None,
                    tool_down=True,
                    yaw_rad=float(yaw),
                    n_restarts=2,
                    max_iter=60,
                )
                apply_q(q_spin)
                _push_q(q_traj_acc, q_spin)
                time.sleep(step_sleep)
            q_now = q_spin.copy()
            apply_q(q_now)
            print(
                f'{tag}原地打转 {twist_deg:.0f}° 完成 '
                f'(last_ik={st_s}/{err_s*1000:.1f}mm)'
            )
            time.sleep(0.25)

            # 仅沿世界 Z 下降 drop_m：锁姿态，避免 6D IK 把 Z 冲过头
            state['last_msg'] = f'{tag}末端下降 Z-{drop_m*100:.0f}cm（忽略点云）…'
            prev_cloud = checker.enable_cloud
            prev_fast = checker.fast_mode
            checker.enable_cloud = False
            checker.fast_mode = True
            try:
                pin.forwardKinematics(model, ik_data, q_now)
                pin.updateFramePlacements(model, ik_data)
                p0 = ik_data.oMf[frame_id].translation.copy()
                z0 = float(p0[2])
                z1 = z0 - drop_m
                q_mid = q_now.copy()
                n_drop = 24
                for k in range(1, n_drop + 1):
                    pk = p0.copy()
                    pk[2] = z0 + (z1 - z0) * (k / n_drop)
                    q_mid, _, _ = ik_position(
                        model,
                        ik_data,
                        frame_id,
                        q_mid,
                        pk,
                        arm_idxs,
                        checker=None,
                        tool_down=False,
                        keep_orientation=True,
                        n_restarts=1,
                        max_iter=40,
                    )
                    apply_q(q_mid)
                    _push_q(q_traj_acc, q_mid)
                    time.sleep(step_sleep)

                # 校正到精确目标 Z（仍保持 XY/姿态）
                pin.forwardKinematics(model, ik_data, q_mid)
                pin.updateFramePlacements(model, ik_data)
                p_now = ik_data.oMf[frame_id].translation.copy()
                if abs(float(p_now[2]) - z1) > 0.005:
                    pk = p_now.copy()
                    pk[2] = z1
                    q_mid, _, _ = ik_position(
                        model,
                        ik_data,
                        frame_id,
                        q_mid,
                        pk,
                        arm_idxs,
                        checker=None,
                        tool_down=False,
                        keep_orientation=True,
                        n_restarts=2,
                        max_iter=60,
                    )
                    apply_q(q_mid)
                    _push_q(q_traj_acc, q_mid)

                pin.forwardKinematics(model, ik_data, q_mid)
                pin.updateFramePlacements(model, ik_data)
                z_end = float(ik_data.oMf[frame_id].translation[2])
                actual_dz = z0 - z_end
                q_now = q_mid.copy()
                apply_q(q_now)
                print(
                    f'{tag}下降完成 目标Δz={drop_m:.3f}m 实测Δz={actual_dz:.3f}m '
                    f'(z {z0:.3f}→{z_end:.3f})'
                )
                time.sleep(0.25)

                # 夹爪示意：合拢合并一下，再张开（写入 grasp_gaps 供 Foxglove）
                state['last_msg'] = f'{tag}夹爪合拢…'
                print(f'{tag}夹爪合拢 → 张开')
                animate_gripper_pinch(
                    viewer,
                    model,
                    ik_data,
                    q_now,
                    frame_id,
                    arm,
                    step_sleep,
                    q_traj_acc=q_traj_acc,
                    grasp_gaps=grasp_gaps,
                )
                state['last_msg'] = f'{tag}夹爪已张开'
                time.sleep(0.15)

                # 关节空间回到到达该轨迹点时的关节角（忽略点云，保证能继续往下走）
                state['last_msg'] = f'{tag}回到途径点…'
                n_up = 28
                for k in range(1, n_up + 1):
                    a = k / n_up
                    qq = (1.0 - a) * q_now + a * q_hold
                    apply_q(qq)
                    _push_q(q_traj_acc, qq)
                    time.sleep(step_sleep)
                q_now = q_hold.copy()
                apply_q(q_now)
                _push_q(q_traj_acc, q_now)
                pin.forwardKinematics(model, ik_data, q_now)
                pin.updateFramePlacements(model, ik_data)
                p_back = ik_data.oMf[frame_id].translation
                print(
                    f'{tag}已回到途径点 '
                    f'xyz=({p_back[0]:.3f},{p_back[1]:.3f},{p_back[2]:.3f})'
                )
            finally:
                checker.enable_cloud = prev_cloud
                checker.fast_mode = prev_fast
            time.sleep(0.3)
            return q_now

        def run():
            try:
                q_cur = state['q'].copy()
                errs = []
                detours = 0
                q_traj: list[list[float]] = [q_cur.tolist()]
                t0 = time.time()
                for ti, tgt in enumerate(targets):
                    state['last_msg'] = f'#{ti+1}/{len(targets)} 求解 IK…'
                    q_goal, err, st = ik_with_detours(
                        model,
                        ik_data,
                        frame_id,
                        q_cur,
                        tgt,
                        arm_idxs,
                        checker,
                        tool_down,
                    )
                    errs.append(err)
                    if st != 'ok' and st != 'ok_via_detour':
                        ok_g, reason_g = checker.check(q_goal)
                        if not ok_g:
                            msg = (
                                f'#{ti+1} 目标位姿不可达/碰障，继续尝试绕障接近: {st}'
                            )
                            print(msg)
                            state['last_msg'] = msg
                            for lift in (0.15, 0.25, 0.35):
                                q_soft, e_soft, st_soft = ik_position(
                                    model,
                                    ik_data,
                                    frame_id,
                                    q_cur,
                                    tgt + np.array([0.0, 0.0, lift]),
                                    arm_idxs,
                                    checker=checker,
                                    tool_down=tool_down,
                                )
                                if st_soft == 'ok':
                                    q_goal, err, st = q_soft, e_soft, 'ok_via_lift'
                                    detours += 1
                                    break
                            if st not in ('ok', 'ok_via_detour', 'ok_via_lift'):
                                print(f'#{ti+1} 跳过，无法无碰到达')
                                state['last_msg'] = f'#{ti+1} 跳过（无法绕开）: {st}'
                                continue

                    if st == 'ok_via_detour':
                        detours += 1

                    state['last_msg'] = f'#{ti+1}/{len(targets)} 规划路径…'
                    t_plan = time.time()
                    path = plan_arm_path(model, q_cur, q_goal, arm_idxs, checker)
                    if path is None:
                        pin.forwardKinematics(model, ik_data, q_cur)
                        pin.updateFramePlacements(model, ik_data)
                        p0 = ik_data.oMf[frame_id].translation.copy()
                        high = np.array(
                            [
                                0.5 * (p0[0] + tgt[0]),
                                0.5 * (p0[1] + tgt[1]),
                                max(p0[2], tgt[2]) + 0.22,
                            ]
                        )
                        q_hi, _, st_hi = ik_position(
                            model,
                            ik_data,
                            frame_id,
                            q_cur,
                            high,
                            arm_idxs,
                            checker=checker,
                            tool_down=tool_down,
                            n_restarts=2,
                        )
                        if st_hi == 'ok':
                            p1 = interpolate_free(q_cur, q_hi, checker, n=8)
                            p2 = interpolate_free(q_hi, q_goal, checker, n=8)
                            if p1 is not None and p2 is not None:
                                path = p1 + p2[1:]
                                detours += 1
                        if path is None:
                            print(f'#{ti+1} 绕障失败，跳过')
                            state['last_msg'] = f'#{ti+1} 绕障失败，已跳过'
                            continue
                    print(f'#{ti+1} 规划 {time.time()-t_plan:.2f}s，{len(path)} 步')

                    play_q_path(path, q_traj)
                    q_cur = q_goal
                    apply_q(q_cur)
                    _push_q(q_traj, q_cur)
                    state['last_msg'] = (
                        f'#{ti+1}/{len(targets)} 已到达，停顿 {dwell_s:.1f}s…'
                    )
                    time.sleep(hold_sleep)
                    if ti < len(do_twist_flags) and do_twist_flags[ti]:
                        q_cur = do_end_twist_and_drop(
                            q_cur,
                            q_traj,
                            q_return=q_goal,
                            tag=f'#{ti+1} ',
                        )

                if return_home:
                    q_cur = go_home_arm(q_cur, q_traj, tag='')

                err_txt = ' → '.join(f'{e*1000:.1f}mm' for e in errs) if errs else '-'
                n_act = sum(1 for f in do_twist_flags if f)
                msg = (
                    f'完成 {arm}臂 {len(targets)} 点；误差 {err_txt}；'
                    f'速度 {speed:.2f}×；绕障次数≈{detours}'
                )
                if n_act:
                    msg += f'；其中 {n_act} 点做了旋转+下降'
                if return_home:
                    msg += '；已复原位'
                print(msg)
                # 自动写入该臂历史（含关节轨迹，可供回放 / Foxglove）
                try:
                    rec = {
                        'arm': arm,
                        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'points': [
                            {
                                'x': float(p[0]),
                                'y': float(p[1]),
                                'z': float(p[2]),
                                'do_twist': bool(
                                    do_twist_flags[i]
                                    if i < len(do_twist_flags)
                                    else False
                                ),
                            }
                            for i, p in enumerate(surface_pts)
                        ],
                        'targets_hover': [
                            {'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])}
                            for p in targets
                        ],
                        'opts': save_opts,
                        'q_traj': q_traj,
                        'grasp_gaps': grasp_gaps,
                        'duration_s': round(time.time() - t0, 3),
                        'note': '执行后自动保存',
                    }
                    hp = save_history_record(arm, rec)
                    state['last_history'] = {
                        'arm': arm,
                        'id': hp.stem,
                        'path': str(hp),
                    }
                    # Foxglove joints_deg 兼容文件（夹合帧全部保留）
                    joints_path = hp.with_name(hp.stem + '_joints_deg.json')
                    step = max(1, len(q_traj) // 200)
                    keep = set(range(0, len(q_traj), step))
                    for i, g in enumerate(grasp_gaps):
                        if g is not None:
                            keep.add(i)
                    poses = []
                    for i in sorted(keep):
                        if i >= len(q_traj):
                            continue
                        pose = {
                            'joints_deg': [
                                float(np.degrees(v)) for v in q_traj[i]
                            ]
                        }
                        if i < len(grasp_gaps) and grasp_gaps[i] is not None:
                            pose['grasp_gap'] = float(grasp_gaps[i])
                        poses.append(pose)
                    joints_path.write_text(
                        json.dumps({'poses': poses}, ensure_ascii=False),
                        encoding='utf-8',
                    )
                    state['last_joints_json'] = str(joints_path)
                    n_g = sum(1 for g in grasp_gaps if g is not None)
                    msg += f'；已存历史 {hp.name}'
                    if n_g:
                        msg += f'（含夹合 {n_g} 帧）'
                except Exception as e:
                    print(f'历史保存失败: {e}')
                state['last_msg'] = msg
            finally:
                state['busy'] = False

        threading.Thread(target=run, daemon=True).start()
        return {
            'ok': True,
            'message': (
                f'开始途经 {len(targets)} 点（上方 {hover_m*100:.0f}cm；'
                f'速度 {speed:.2f}×；到点停顿 {dwell_s:.1f}s；'
                f'旋转+下降点数 {sum(1 for f in do_twist_flags if f)}；'
                f'结束后{"复原位" if return_home else "不复原"}）'
            ),
        }

    def animate_home(arm: str, opts: dict | None = None) -> dict:
        opts = opts or {}
        if state['busy']:
            return {'ok': False, 'message': '正在运动，请稍候'}
        checker.set_params(
            inflate_self_m=float(opts.get('inflate_self_m', 0.02)),
            inflate_cloud_m=float(opts.get('inflate_cloud_m', 0.03)),
            enable_self=bool(opts.get('avoid_self', True)),
            enable_cloud=bool(opts.get('avoid_cloud', True)),
            active_arm=arm,
        )
        arm_idxs = arm_q_indices(model, arm)
        if len(arm_idxs) < 7:
            return {'ok': False, 'message': f'关节索引异常: {arm_idxs}'}
        speed = max(0.05, min(5.0, float(opts.get('speed', 0.35))))
        step_sleep = 0.035 / speed
        state['busy'] = True

        def run():
            try:
                q_cur = state['q'].copy()
                q_goal = q_cur.copy()
                q_n = pin.neutral(model)
                for ji in arm_idxs:
                    q_goal[ji] = float(q_n[ji])
                q_goal = clamp_q(model, q_goal)
                state['last_msg'] = f'{arm}臂复原位…'
                path = plan_arm_path(model, q_cur, q_goal, arm_idxs, checker)
                if path is None:
                    prev_cloud = checker.enable_cloud
                    checker.enable_cloud = False
                    try:
                        path = plan_arm_path(model, q_cur, q_goal, arm_idxs, checker)
                    finally:
                        checker.enable_cloud = prev_cloud
                if path is None:
                    path = interpolate_free(q_cur, q_goal, None, n=24) or [
                        q_cur + (q_goal - q_cur) * (k / 24.0) for k in range(1, 25)
                    ]
                    path[-1] = q_goal
                for qq in path:
                    apply_q(qq)
                    time.sleep(step_sleep)
                apply_q(q_goal)
                msg = f'{arm}臂已复原位'
                print(msg)
                state['last_msg'] = msg
            finally:
                state['busy'] = False

        threading.Thread(target=run, daemon=True).start()
        return {'ok': True, 'message': f'开始 {arm}臂复原位'}

    def replay_history(arm: str, hist_id: str, speed: float = 0.35, dwell_s: float = 2.0) -> dict:
        if state['busy']:
            return {'ok': False, 'message': '正在运动，请稍候'}
        rec = load_history_record(arm, hist_id)
        if rec is None:
            return {'ok': False, 'message': f'找不到历史 {hist_id}'}
        q_traj = rec.get('q_traj') or []
        if len(q_traj) < 2:
            pts = rec.get('points') or []
            opts = dict(rec.get('opts') or {})
            opts['speed'] = speed
            opts['dwell_s'] = dwell_s
            return animate_goto(arm, pts, opts)
        speed = max(0.05, min(5.0, float(speed)))
        dwell_s = max(0.0, min(30.0, float(dwell_s)))
        step_sleep = 0.035 / speed
        state['busy'] = True
        clear_pick_viz(viewer)

        def run():
            try:
                state['last_msg'] = f'回放 {arm}臂历史 {hist_id}…'
                n = len(q_traj)
                # 粗略按途经点均分停顿：有 points 则用点数，否则每 1/4 轨迹停一次
                n_wpt = max(1, len(rec.get('points') or []) or 4)
                pause_every = max(1, n // n_wpt)
                for i, qrow in enumerate(q_traj):
                    apply_q(np.asarray(qrow, dtype=float))
                    if i % 8 == 0:
                        state['last_msg'] = f'回放 {i+1}/{n}'
                    time.sleep(step_sleep)
                    if pause_every > 1 and i > 0 and (i + 1) % pause_every == 0 and i + 1 < n:
                        state['last_msg'] = f'回放途经点停顿 {dwell_s:.1f}s…'
                        time.sleep(dwell_s)
                state['last_msg'] = f'回放完成 {hist_id}（{n} 帧）'
            finally:
                state['busy'] = False

        threading.Thread(target=run, daemon=True).start()
        return {
            'ok': True,
            'message': f'开始回放 {hist_id}（{len(q_traj)} 帧；到点停顿 {dwell_s:.1f}s）',
        }
    def export_foxglove_mcap(arm: str) -> dict:
        """用最近一次关节日志调用 record_foxglove_mcap.py 生成可在 Foxglove 播放的 mcap。"""
        import subprocess

        joints = state.get('last_joints_json')
        if not joints or not Path(joints).is_file():
            # fallback: newest history for this arm
            items = list_history(arm)
            if not items:
                return {'ok': False, 'message': '没有可导出的历史，请先执行一次轨迹'}
            jpath = history_dir(arm) / f"{items[0]['id']}_joints_deg.json"
            if not jpath.is_file():
                return {
                    'ok': False,
                    'message': '历史无关节采样（旧记录）。请重新执行一次轨迹后再导出',
                }
            joints = str(jpath)
        FOXGLOVE_OUT.mkdir(parents=True, exist_ok=True)
        out = FOXGLOVE_OUT / f'demo_arm{arm}_{time.strftime("%Y%m%d_%H%M%S")}.mcap'
        script = Path(__file__).resolve().parent / 'record_foxglove_mcap.py'
        py = '/home/ubuntu/miniconda3/envs/teleop/bin/python'
        cmd = [
            py,
            str(script),
            '--urdf',
            str(args.urdf),
            '--joints-json',
            joints,
            '--out',
            str(out),
            '--drive-arm',
            'armL' if arm == 'L' else 'armR',
            '--dt',
            '0.15',
            '--cloud-x-offset-m',
            str(args.cloud_x_offset_m),
            '--cloud-y-offset-m',
            str(args.cloud_y_offset_m),
            '--cloud-z-offset-m',
            str(args.cloud_z_offset_m),
        ]
        if args.ply.is_file():
            cmd.extend(['--ply', str(args.ply)])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                err = (r.stderr or r.stdout or '').strip()
                if len(err) > 600:
                    err = err[-600:]
                return {
                    'ok': False,
                    'message': f'导出失败 (exit {r.returncode}): {err or "无详细输出"}',
                }
        except Exception as e:
            return {'ok': False, 'message': f'导出失败: {e}'}
        return {
            'ok': True,
            'message': (
                f'已生成 {out}。Foxglove 打开后：3D 勾 /arm 与 /gripper（夹合为末端红块+黄点，'
                f'拖时间轴到中段下降处可见）。关节源: {joints}'
            ),
            'mcap': str(out),
            'joints': joints,
        }
    class IndexHandler(tornado.web.RequestHandler):
        def get(self):
            self.set_header('Cache-Control', 'no-store')
            self.set_header('Content-Type', 'text/html; charset=utf-8')
            self.write(UI_HTML)

    class ApiCloudHandler(tornado.web.RequestHandler):
        def get(self):
            self.set_header('Cache-Control', 'no-store')
            self.write(cloud_ui)

    class ApiStatusHandler(tornado.web.RequestHandler):
        def get(self):
            self.set_header('Cache-Control', 'no-store')
            self.write(
                {
                    'busy': bool(state.get('busy')),
                    'message': state.get('last_msg') or ('运动中…' if state.get('busy') else '就绪'),
                }
            )

    class ApiHistoryListHandler(tornado.web.RequestHandler):
        def get(self):
            arm = self.get_argument('arm', 'L')
            self.set_header('Cache-Control', 'no-store')
            self.write({'items': list_history(arm)})

    class ApiJsonHandler(tornado.web.RequestHandler):
        def prepare(self):
            raw = self.request.body or b'{}'
            try:
                self.json = json.loads(raw.decode('utf-8') or '{}')
            except Exception:
                self.json = {}

        def post(self):
            path = self.request.path
            payload = self.json
            if path == '/api/q':
                qq = state['q'].copy()
                for k, v in (payload.get('q') or {}).items():
                    qq[int(k)] = float(v)
                apply_q(qq)
                self.set_status(204)
                return
            if path == '/api/snap':
                z = payload.get('z')
                self.write(
                    snap_xy(
                        float(payload['x']),
                        float(payload['y']),
                        None if z is None else float(z),
                    )
                )
                return
            if path == '/api/markers':
                # 演示 3D 不再画标记/轨迹线
                clear_pick_viz(viewer)
                self.write({'ok': True})
                return
            if path == '/api/clear_markers':
                clear_pick_viz(viewer)
                self.write({'ok': True})
                return
            if path == '/api/goto':
                arm = payload.get('arm', 'L')
                pts = payload.get('points') or []
                self.write(animate_goto(arm, pts, payload))
                return
            if path == '/api/home':
                self.write(animate_home(payload.get('arm', 'L'), payload))
                return
            if path == '/api/history/save':
                arm = payload.get('arm', 'L')
                pts = payload.get('points') or []
                if not pts:
                    self.write({'ok': False, 'message': '没有途径点'})
                    return
                rec = {
                    'arm': arm,
                    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'points': pts,
                    'opts': payload.get('opts') or {},
                    'q_traj': [],
                    'note': payload.get('note') or '手动保存途径点',
                }
                hp = save_history_record(arm, rec)
                self.write({'ok': True, 'message': f'已保存 {hp.name}', 'id': hp.stem})
                return
            if path == '/api/history/load':
                arm = payload.get('arm', 'L')
                hist_id = payload.get('id', '')
                rec = load_history_record(arm, hist_id)
                if rec is None:
                    self.write({'ok': False, 'message': f'找不到 {hist_id}'})
                    return
                self.write(
                    {
                        'ok': True,
                        'points': rec.get('points') or [],
                        'opts': rec.get('opts') or {},
                        'id': hist_id,
                    }
                )
                return
            if path == '/api/history/delete':
                arm = payload.get('arm', 'L')
                hist_id = payload.get('id', '')
                path_h = history_dir(arm) / f'{hist_id}.json'
                side = history_dir(arm) / f'{hist_id}_joints_deg.json'
                if path_h.is_file():
                    path_h.unlink()
                if side.is_file():
                    side.unlink()
                self.write({'ok': True, 'message': f'已删除 {hist_id}'})
                return
            if path == '/api/history/replay':
                self.write(
                    replay_history(
                        payload.get('arm', 'L'),
                        payload.get('id', ''),
                        float(payload.get('speed', 0.35)),
                        float(payload.get('dwell_s', 2.0)),
                    )
                )
                return
            if path == '/api/export_foxglove':
                self.write(export_foxglove_mcap(payload.get('arm', 'L')))
                return
            self.set_status(404)

    class MeshcatWsProxy(tornado.websocket.WebSocketHandler):
        """Bridge browser WS on :8765 to Meshcat Tornado on :7000."""

        def check_origin(self, origin):
            return True

        async def open(self):
            self.upstream = None
            mesh_url = viewer.url()  # http://127.0.0.1:7000/static/
            u = urlparse(mesh_url)
            up_ws = f'ws://{u.hostname}:{u.port}/'

            def on_message(msg):
                if self.ws_connection is None:
                    return
                try:
                    if isinstance(msg, (bytes, bytearray)):
                        self.write_message(msg, binary=True)
                    else:
                        self.write_message(msg)
                except Exception:
                    pass

            try:
                self.upstream = await websocket_connect(
                    up_ws, on_message_callback=on_message
                )
            except Exception as e:
                print(f'meshcat ws proxy open failed: {e}')
                self.close()

        def on_message(self, message):
            if self.upstream is None:
                return
            try:
                self.upstream.write_message(message)
            except Exception:
                pass

        def on_close(self):
            if self.upstream is not None:
                try:
                    self.upstream.close()
                except Exception:
                    pass
                self.upstream = None

    app = tornado.web.Application(
        [
            (r'/', IndexHandler),
            (r'/api/cloud', ApiCloudHandler),
            (r'/api/status', ApiStatusHandler),
            (r'/api/history', ApiHistoryListHandler),
            (
                r'/api/(?:q|snap|markers|clear_markers|goto|history/save|history/load|history/delete|history/replay|export_foxglove)',
                ApiJsonHandler,
            ),
            (r'/mc/ws', MeshcatWsProxy),
            (
                r'/mc/(.*)',
                tornado.web.StaticFileHandler,
                {'path': str(VIEWER_ROOT), 'default_filename': 'index.html'},
            ),
        ],
        compress_response=False,
    )
    app.listen(args.ui_port, address=args.host)
    print('=' * 60)
    print('合并演示页: ', f'http://127.0.0.1:{args.ui_port}/')
    print('（左控制 + 右 3D 同页；转发 8765 即可，7000/6000 本机保留）')
    print('=' * 60)
    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print('bye')


if __name__ == '__main__':
    main()
