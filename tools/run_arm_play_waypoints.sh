#!/usr/bin/env bash
# 回放录制路点（rm_movel 匀速直线）。默认 dry-run；加 --execute 才动。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CALIBRATION_ROOT="${CALIBRATION_ROOT:-/home/ubuntu/workspace/lk_ceshi/Calibration}"
ARM_IP="${ARM_IP:-192.168.1.18}"
WP="${WAYPOINTS:-$ROOT/gpu_logs/scan_waypoints_latest.json}"
exec python3 "$ROOT/tools/arm_play_waypoints.py" \
  --arm-ip "$ARM_IP" \
  --waypoints "$WP" \
  --speed "${SPEED:-15}" \
  --dwell "${DWELL:-0.3}" \
  "$@"
