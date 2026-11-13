#!/usr/bin/env bash
# 实验 G：URDF 连杆 AABB 膨胀 5cm，剔除自身点云
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CALIBRATION_ROOT="${CALIBRATION_ROOT:-/home/ubuntu/workspace/lk_ceshi/Calibration}"
URDF="${URDF:-/home/ubuntu/workspace/lk_ceshi/robot_models/rm_models_RM75/urdf/RM75-B/urdf/RM75-B.urdf}"
PLY="${PLY:-$ROOT/live_mesh.ply}"
OUT="${OUT:-$ROOT/live_mesh_self_filtered.ply}"
INFLATE_M="${INFLATE_M:-0.05}"

ARGS=(
  --urdf "$URDF"
  --ply "$PLY"
  --out "$OUT"
  --inflate-m "$INFLATE_M"
)

if [[ -n "${JOINTS_JSON:-}" ]]; then
  ARGS+=(--joints-json "$JOINTS_JSON")
elif [[ -n "${JOINTS_DEG:-}" ]]; then
  ARGS+=(--joints-deg "$JOINTS_DEG")
else
  ARGS+=(--from-arm --arm-ip "${ARM_IP:-192.168.1.18}")
fi

if [[ -n "${DUMP_BOXES:-}" ]]; then
  ARGS+=(--dump-boxes-json "$DUMP_BOXES")
fi

exec python3 "$ROOT/tools/arm_self_filter_ply.py" "${ARGS[@]}" "$@"
