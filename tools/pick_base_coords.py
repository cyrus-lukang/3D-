#!/usr/bin/env python3
"""在重建网格上点选，显示机械臂 Base 坐标系下的坐标。

臂位姿建图（--pose_source arm）时 nvblox 世界系 = RealMan Base：
  T_W_C = T_base_ee @ T_tool_cam
因此 mesh 顶点坐标可直接当作 Base 下 (x,y,z)，单位米。

用法（本机有显示时）：
  /home/ubuntu/miniconda3/envs/gsplat/bin/python \\
    tools/pick_base_coords.py --mesh live_mesh.ply

操作：
  Shift + 左键  点选一个顶点（可多点）
  Q / Esc       关闭窗口
关闭后终端打印各点 Base xyz，并可选写入 CSV 方便与示教/尺规真值对比。
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pick mesh points → Base XYZ")
    p.add_argument(
        "--mesh",
        type=str,
        default="live_mesh.ply",
        help="nvblox 导出的彩色网格（默认 live_mesh.ply）",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="保存点选 CSV 路径；默认 gpu_logs/picked_base_YYYYMMDD_HHMMSS.csv",
    )
    p.add_argument(
        "--voxel",
        type=float,
        default=0.0,
        help="可选：下采样体素(m)，大网格卡顿时用 0.005~0.01",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "需要 open3d。可用: /home/ubuntu/miniconda3/envs/gsplat/bin/python"
        ) from exc

    root = Path(__file__).resolve().parents[1]
    mesh_path = Path(args.mesh)
    if not mesh_path.is_absolute():
        mesh_path = root / mesh_path
    if not mesh_path.is_file():
        raise SystemExit(f"找不到 mesh: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        # 有些导出其实是点云 PLY
        pcd = o3d.io.read_point_cloud(str(mesh_path))
        if pcd.is_empty():
            raise SystemExit(f"空网格/点云: {mesh_path}")
        geom = pcd
        print(f"加载点云 {mesh_path}  points={np.asarray(pcd.points).shape[0]}")
    else:
        mesh.compute_vertex_normals()
        if args.voxel > 0:
            pcd = mesh.sample_points_uniformly(
                number_of_points=max(50000, int(1.0 / (args.voxel**2)))
            )
            pcd = pcd.voxel_down_sample(args.voxel)
            geom = pcd
            print(f"mesh→下采样点云 voxel={args.voxel}m  n={len(pcd.points)}")
        else:
            # 点选工具对 PointCloud 更稳；从顶点采样
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            if mesh.has_vertex_colors():
                pcd.colors = mesh.vertex_colors
            geom = pcd
            print(f"加载网格顶点 {mesh_path}  n={len(pcd.points)}")

    # Base 坐标轴（红X 绿Y 蓝Z），原点画在点云质心附近仅作参考时可选；
    # 这里画在世界原点，方便确认 Base 原点相对场景的位置。
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    print(
        "\n=== 点选说明 ===\n"
        "  Shift + 左键 : 选点（可连选多个）\n"
        "  Q / Esc      : 结束并打印坐标\n"
        "单位: 米 (Base 系)。与示教器对比时注意单位换算。\n"
    )

    # VisualizerWithEditing: Shift+click 选点，关闭后返回点索引
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick Base XYZ (Shift+LMB)", width=1280, height=800)
    vis.add_geometry(geom)
    vis.add_geometry(axis)
    vis.run()  # 阻塞直到关闭
    vis.destroy_window()
    idxs = vis.get_picked_points()

    pts = np.asarray(geom.points)
    if not idxs:
        print("未选中任何点。")
        return 0

    print("\n#  |     X(m)      Y(m)      Z(m)  |    X(mm)     Y(mm)     Z(mm)")
    print("-" * 72)
    rows = []
    for i, idx in enumerate(idxs, 1):
        if idx < 0 or idx >= len(pts):
            continue
        x, y, z = pts[idx]
        print(
            f"{i:2d} | {x:9.4f} {y:9.4f} {z:9.4f} | "
            f"{x*1000:9.1f} {y*1000:9.1f} {z*1000:9.1f}"
        )
        rows.append(
            {
                "index": i,
                "vertex_id": int(idx),
                "x_m": float(x),
                "y_m": float(y),
                "z_m": float(z),
                "x_mm": float(x * 1000),
                "y_mm": float(y * 1000),
                "z_mm": float(z * 1000),
                "note_gt_x_mm": "",
                "note_gt_y_mm": "",
                "note_gt_z_mm": "",
                "err_mm": "",
            }
        )

    # 相邻点距（便于量托盘边长）
    if len(rows) >= 2:
        print("\n相邻点距离:")
        for a, b in zip(rows, rows[1:]):
            d = np.linalg.norm(
                [
                    b["x_m"] - a["x_m"],
                    b["y_m"] - a["y_m"],
                    b["z_m"] - a["z_m"],
                ]
            )
            print(f"  P{a['index']}→P{b['index']}: {d*1000:.1f} mm  ({d:.4f} m)")

    out = Path(args.out) if args.out else (
        root / "gpu_logs" / f"picked_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写入 {out}")
    print("把示教器/尺规真值填到 note_gt_*_mm，再算误差即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
