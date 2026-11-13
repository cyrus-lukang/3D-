#!/usr/bin/env bash
# 录制 N 个 EE 位姿（默认 5）。示教器挪好 → 回车记录。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CALIBRATION_ROOT="${CALIBRATION_ROOT:-/home/ubuntu/workspace/lk_ceshi/Calibration}"
ARM_IP="${ARM_IP:-192.168.1.18}"
exec python3 "$ROOT/tools/arm_record_waypoints.py" \
  --arm-ip "$ARM_IP" \
  --count "${COUNT:-5}" \
  "$@"
