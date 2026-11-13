#!/usr/bin/env bash
# 其它关节不动，只转末端关节（默认 J7）180°。
# 默认 dry-run；真正动臂：加 --execute，并输入 YES。
#
# 例：
#   bash tools/run_arm_rotate_ee.sh
#   bash tools/run_arm_rotate_ee.sh --execute
#   DEGREES=-180 bash tools/run_arm_rotate_ee.sh --return --execute
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CALIBRATION_ROOT="${CALIBRATION_ROOT:-/home/ubuntu/workspace/lk_ceshi/Calibration}"
ARM_IP="${ARM_IP:-192.168.1.18}"

exec python3 "$ROOT/tools/arm_rotate_ee_joint.py" \
  --arm-ip "$ARM_IP" \
  --joint "${JOINT:-7}" \
  --degrees "${DEGREES:-180}" \
  --speed "${SPEED:-8}" \
  --dwell "${DWELL:-0.5}" \
  "$@"
