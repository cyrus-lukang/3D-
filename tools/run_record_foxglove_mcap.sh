#!/usr/bin/env bash
# 离线打包 Foxglove MCAP（彩色近完整顶点地图 + TF + URDF）
# 渐进：PROGRESSIVE_DIR=gpu_logs/progressive_map ./tools/run_record_foxglove_mcap.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOINTS_JSON="${JOINTS_JSON:-$ROOT/samples/scan_joints.json}"
URDF="${URDF:-$ROOT/rimbot-TJ-urdf_260822-6/urdf/rimbot-TJ-urdf_260822-6.urdf}"
# 0 = near-full vertices (no downsample); set MAX_POINTS=200000 to shrink
MAX_POINTS="${MAX_POINTS:-0}"
# 7DOF 日志驱动双臂 URDF 的哪一侧：armR | armL | both
DRIVE_ARM="${DRIVE_ARM:-armR}"
MESH_MODE="${MESH_MODE:-core}"
# 基座系固定高度参考面（米）。例: HEIGHT_PLANE_Z=0 或 0.75
HEIGHT_PLANE_Z="${HEIGHT_PLANE_Z:-}"
# 点云整体沿 base_link Z 上升（米），例: CLOUD_Z_OFFSET_M=0.75
CLOUD_Z_OFFSET_M="${CLOUD_Z_OFFSET_M:-0}"
CLOUD_X_OFFSET_M="${CLOUD_X_OFFSET_M:-0}"
CLOUD_Y_OFFSET_M="${CLOUD_Y_OFFSET_M:-0}"

ARGS=(
  --joints-json "$JOINTS_JSON"
  --urdf "$URDF"
  --max-points "$MAX_POINTS"
  --drive-arm "$DRIVE_ARM"
  --mesh-mode "$MESH_MODE"
  --cloud-x-offset-m "$CLOUD_X_OFFSET_M"
  --cloud-y-offset-m "$CLOUD_Y_OFFSET_M"
  --cloud-z-offset-m "$CLOUD_Z_OFFSET_M"
)
if [[ -n "$HEIGHT_PLANE_Z" ]]; then
  ARGS+=(--height-plane-z "$HEIGHT_PLANE_Z")
fi
if [[ -n "${PROGRESSIVE_DIR:-}" ]]; then
  OUT="${OUT:-$ROOT/gpu_logs/session_progressive.mcap}"
  ARGS+=(--out "$OUT" --progressive-dir "$PROGRESSIVE_DIR")
else
  OUT="${OUT:-$ROOT/samples/session_foxglove.mcap}"
  PLY="${PLY:-$ROOT/samples/live_mesh_self_filtered.ply}"
  if [[ ! -f "$PLY" ]]; then
    PLY="$ROOT/samples/live_mesh_self_filtered.ply"
  fi
  ARGS+=(--out "$OUT" --ply "$PLY")
fi

exec python3 "$ROOT/tools/record_foxglove_mcap.py" "${ARGS[@]}" "$@"
