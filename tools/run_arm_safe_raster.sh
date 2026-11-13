#!/usr/bin/env bash
# RealMan 安全盒弓字扫视（给 nvblox 建图用）
# 默认 dry-run：只打印路径。真正动臂：加 --execute，并输入 YES。
#
# 推荐：终端1 先跑 mapper；终端2 再跑本脚本。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CALIBRATION_ROOT="${CALIBRATION_ROOT:-/home/ubuntu/workspace/lk_ceshi/Calibration}"
ARM_IP="${ARM_IP:-192.168.1.18}"

exec python3 "$ROOT/tools/arm_safe_raster_scan.py" \
  --arm-ip "$ARM_IP" \
  --box-from-current \
  --dx "${DX:-0.06}" \
  --dy "${DY:-0.08}" \
  --dz "${DZ:-0.0}" \
  --step "${STEP:-0.04}" \
  --speed "${SPEED:-10}" \
  --dwell "${DWELL:-0.4}" \
  --pitch-offsets-deg "${PITCH_DEG:-0}" \
  "$@"
