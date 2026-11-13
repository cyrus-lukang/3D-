#!/usr/bin/env bash
# Meshcat URDF 交互演示（控制页 + 3D）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
pkill -f view_urdf_meshcat.py 2>/dev/null || true
pkill -f 'meshcat.servers.zmqserver' 2>/dev/null || true
sleep 0.5
fuser -k 7000/tcp 7001/tcp 6000/tcp 8765/tcp 2>/dev/null || true
sleep 0.3
export PYTHONUNBUFFERED=1
URDF="${1:-$ROOT/rimbot-TJ-urdf_260822-6/urdf/rimbot-TJ-urdf_260822-6.urdf}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "启动合并演示页（左控制 + 右 3D 同页）…"
echo "浏览器打开: http://127.0.0.1:8765/"
exec "$PYTHON_BIN" -u "$ROOT/tools/view_urdf_meshcat.py" --urdf "$URDF" --ui-port 8765
