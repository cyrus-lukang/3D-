# 本机 Agent 交接：PLY 点选 Base 坐标 vs 真值

> 给**本机有 GUI 的 Agent**用。当前任务只做离线精度核对，**不要**装 nvblox / CUDA / RealSense / FoundationStereo 全套重建环境。

## 我在做什么

1. 在 4090 服务器上用 **机械臂位姿（`--pose_source arm`）+ FoundationStereo 深度** 跑 nvblox 建图。
2. 此时 **nvblox 世界系 = RealMan Base**（`T_W_C = T_base_ee @ T_tool_cam`），导出的 mesh 顶点坐标就是 Base 下 `(x,y,z)`，单位**米**。
3. 把 `live_mesh.ply`（或其它导出 PLY）拷到本机，用 `tools/pick_base_coords.py` **Shift+左键点选**特征点（二维码角点、托盘角等）。
4. 把点选结果与**示教器 / 尺规真值**对比，算误差（mm）。

仓库：`https://github.com/cyrus-lukang/nvblox`（分支 `main` / `nvblox_3D`）。核心脚本：`tools/pick_base_coords.py`。

## 本机只要装这些

最小 Python 环境（有显示器；Open3D 需要本机窗口）：

```bash
# 建议独立 venv / conda
python3 -m venv .venv-pick
source .venv-pick/bin/activate   # Windows: .venv-pick\Scripts\activate

pip install -U pip
pip install "numpy" "open3d>=0.18,<=0.19"
```

- **需要**：`numpy`、`open3d`（带 GUI 的 VisualizerWithEditing）
- **不需要**：nvblox 编译、torch、CUDA、realsense、飞书脚本、建图依赖

验证：

```bash
python -c "import open3d as o3d, numpy; print(o3d.__version__, 'ok')"
```

## 准备 PLY

把服务器上的网格拷到本机仓库根目录（或任意路径）：

```bash
# 示例：从 10.3.1.10 拷（按你本机 ssh 配置改）
scp user@10.3.1.10:/path/to/nvblox_3D重建/live_mesh.ply ./live_mesh.ply
```

常见文件名：`live_mesh.ply`。也可以是点云 PLY；脚本会自动 mesh→顶点 或读点云。

大网格卡顿时加 `--voxel 0.005`～`0.01`。

## 怎么跑

在仓库根目录：

```bash
python tools/pick_base_coords.py --mesh live_mesh.ply
# 或
python tools/pick_base_coords.py --mesh /绝对路径/live_mesh.ply --out gpu_logs/picked_base.csv
```

操作：

| 键鼠 | 作用 |
|------|------|
| **Shift + 左键** | 选点（可多选） |
| **Q / Esc** | 关窗；终端打印 Base xyz |

窗口里红/绿/蓝轴是 Base 原点处的 XYZ（0.1 m）。

关闭后会：

1. 打印每点 `m` 与 `mm`
2. 打印相邻点距离（方便量边长）
3. 写 CSV（默认 `gpu_logs/picked_base_YYYYMMDD_HHMMSS.csv`）

CSV 列：`x_m,y_m,z_m,x_mm,y_mm,z_mm` + 空列 `note_gt_x_mm,note_gt_y_mm,note_gt_z_mm,err_mm`。

## 与真值对比（Agent 要做的）

1. 用户从示教器/尺规给出真值（一般是 **mm**，注意与米单位换算）。
2. 填入 CSV 的 `note_gt_*_mm`，或直接在终端对比。
3. 误差建议：

\[
e = \sqrt{(x-x_{gt})^2 + (y-y_{gt})^2 + (z-z_{gt})^2}
\]

单位统一成 mm 后再算；可写回 `err_mm`。

示例（填完真值后）：

```python
import csv, math
from pathlib import Path
p = Path("gpu_logs/picked_base_XXXX.csv")  # 改成实际文件
rows = list(csv.DictReader(p.open(encoding="utf-8")))
for r in rows:
    if not r["note_gt_x_mm"]:
        continue
    dx = float(r["x_mm"]) - float(r["note_gt_x_mm"])
    dy = float(r["y_mm"]) - float(r["note_gt_y_mm"])
    dz = float(r["z_mm"]) - float(r["note_gt_z_mm"])
    r["err_mm"] = f"{math.sqrt(dx*dx+dy*dy+dz*dz):.2f}"
    print(r["index"], "err_mm=", r["err_mm"])
```

## Agent 行为约束

- **目标**：装最小环境 → 打开 PLY → 点选 / 协助对比真值；不要去编译 nvblox 或拉重建数据集。
- 必须在有图形界面的本机跑；SSH 无显示会失败。
- 坐标系：点选结果已是 **Base**；不要再乘手眼矩阵。
- 示教器若用 mm，对比时用 CSV 的 `*_mm` 列。

## 相关但不在本任务

服务器侧建图、FFS、飞书文档等见同目录其它脚本（`run_mapper_arm_ffs.sh` 等）。本机 Agent **跳过**。
