#!/usr/bin/env bash
# Needs a real DISPLAY and: ros-humble-joint-state-publisher-gui
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /opt/ros/humble/setup.bash
URDF="${1:-$ROOT/rimbot-TJ-urdf_260822-6/urdf/rimbot-TJ-abs.urdf}"
if [[ ! -f "$URDF" ]]; then
  python3 - <<PY
from pathlib import Path
import re
u=Path("$ROOT/rimbot-TJ-urdf_260822-6/urdf/rimbot-TJ-urdf_260822-6.urdf")
pkg=u.resolve().parents[1]
t=re.sub(r"package://[^/]+/", str(pkg)+"/", u.read_text())
Path("$URDF").write_text(t)
print("wrote $URDF")
PY
fi
DESC=$(cat "$URDF")
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$DESC" &
RSP=$!
if ros2 pkg prefix joint_state_publisher_gui >/dev/null 2>&1; then
  ros2 run joint_state_publisher_gui joint_state_publisher_gui &
else
  echo "WARN: joint_state_publisher_gui 未安装，只有零位姿态"
  ros2 run joint_state_publisher joint_state_publisher 2>/dev/null &
fi
JSP=$!
rviz2 -d $ROOT/tools/urdf_view.rviz 2>/dev/null || rviz2
wait
