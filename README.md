# 3D Demo — 三维重建演示 / URDF 运动

演示主线：头相机彩色点云/网格 + Foxglove / Meshcat 里叠 **rimbot-TJ URDF**，按关节轨迹回放运动。

本仓库只保留演示建模与运动相关内容（不含完整 nvblox 工程与大数据）。

## 目录

| 路径 | 说明 |
| --- | --- |
| `rimbot-TJ-urdf_260822-6/` | 双臂 URDF + meshes |
| `tools/` | Meshcat / Foxglove / 路点录放与扫掠 |
| `traj_history/` | 已录关节/路点轨迹 |
| `samples/` | 示例网格、关节 JSON、演示 MCAP |
| `docs/` | 演示简报与建图说明 |

## 快速开始

### 1) Meshcat：URDF + 点云交互

```bash
./tools/run_urdf_viewer.sh
# 浏览器打开 http://127.0.0.1:8765/
```

依赖：`pinocchio`、`meshcat`、`numpy`（可用已有 `teleop` 环境：`PYTHON_BIN=... ./tools/run_urdf_viewer.sh`）。

### 2) 打包 Foxglove MCAP（点云/网格 + URDF TF + 关节）

```bash
./tools/run_record_foxglove_mcap.sh
# 或
python3 tools/record_foxglove_mcap.py \
  --ply samples/live_mesh_self_filtered.ply \
  --joints-json samples/scan_joints.json \
  --urdf rimbot-TJ-urdf_260822-6/urdf/rimbot-TJ-urdf_260822-6.urdf \
  --out samples/session_foxglove.mcap
```

Foxglove Studio → Open local file → 打开 `samples/*.mcap`。仓库已带示例：`samples/demo_armL_with_gripper.mcap`。

### 3) 路点录制 / 回放（真机）

需本机标定与机械臂 SDK 环境：

```bash
export CALIBRATION_ROOT=/path/to/Calibration
./tools/run_arm_record_waypoints.sh
./tools/run_arm_play_waypoints.sh
```

## 说明

- 演示用途是「场景长什么样、臂和地图是否对齐」，不是避障验收。
- 更完整建图链路仍在原 `nvblox_3D重建` 工程中。
