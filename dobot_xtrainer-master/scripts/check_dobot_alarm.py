"""Read Dobot controller modes and optionally clear a selected alarm.

Without ``--clear`` this script is read-only: it only sends ``RobotMode()`` and
``GetErrorID()``.  Clearing is deliberately opt-in because the operator must
first remove collisions, release emergency stops, and make the cell safe.
"""

import argparse
import ast
import dataclasses
import os
import re
import sys
import time
from typing import Iterable, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dobot_control.robots.dobot_api import DobotApiDashboard


CONTROLLERS = {
    "left": "192.168.5.1",
    "right": "192.168.5.2",
}

MODE_NAMES = {
    1: "INIT",
    2: "BRAKE_OPEN",
    3: "POWEROFF",
    4: "DISABLED",
    5: "ENABLED_IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "SINGLE_MOVE",
    9: "ERROR",
    10: "PAUSE",
    11: "COLLISION",
}


@dataclasses.dataclass(frozen=True)
class ControllerStatus:
    name: str
    ip: str
    mode: Optional[int]
    error_ids: List[int]
    mode_response: str
    error_response: str

    @property
    def alarmed(self) -> bool:
        return self.mode in (9, 11)


def parse_robot_mode(response: str) -> Optional[int]:
    match = re.search(r"\{\s*(-?\d+)\s*\}", response or "")
    return int(match.group(1)) if match else None


def _flatten_ints(value) -> Iterable[int]:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_ints(item)


def parse_error_ids(response: str) -> List[int]:
    if not response:
        return []
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        payload = ast.literal_eval(response[start + 1 : end].strip())
    except (SyntaxError, ValueError):
        return []
    return list(_flatten_ints(payload))


def query_controller(name: str, ip: str) -> ControllerStatus:
    dashboard = DobotApiDashboard(ip, 29999)
    try:
        mode_response = dashboard.RobotMode()
        error_response = dashboard.GetErrorID()
    finally:
        dashboard.close()
    return ControllerStatus(
        name=name,
        ip=ip,
        mode=parse_robot_mode(mode_response),
        error_ids=parse_error_ids(error_response),
        mode_response=mode_response.strip(),
        error_response=error_response.strip(),
    )


def query_all_controllers() -> List[ControllerStatus]:
    return [query_controller(name, ip) for name, ip in CONTROLLERS.items()]


def format_status(status: ControllerStatus) -> str:
    mode_name = MODE_NAMES.get(status.mode, "UNKNOWN")
    errors = status.error_ids or []
    suffix = ""
    if errors == [-2]:
        suffix = " (-2 is the generic 'robot is alarmed' return, not a joint ID)"
    return (
        f"{status.name}: {status.ip} mode={status.mode}({mode_name}) "
        f"errors={errors}{suffix}"
    )


def clear_controller(name: str) -> str:
    ip = CONTROLLERS[name]
    dashboard = DobotApiDashboard(ip, 29999)
    try:
        return dashboard.ClearError().strip()
    finally:
        dashboard.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read Dobot alarms; clearing requires an explicit --clear target.",
    )
    parser.add_argument(
        "--clear",
        choices=("left", "right", "both"),
        help="Clear the selected controller alarm after the operator checks the cell.",
    )
    args = parser.parse_args()

    before = query_all_controllers()
    for status in before:
        print(format_status(status))

    if args.clear:
        targets = ("left", "right") if args.clear == "both" else (args.clear,)
        for name in targets:
            print(f"clear {name}: {clear_controller(name)}")
        time.sleep(0.5)
        print("after clear:")
        after = query_all_controllers()
        for status in after:
            print(format_status(status))
        return int(any(status.alarmed for status in after))

    if any(status.alarmed for status in before):
        print(
            "Alarm remains. Inspect the physical cell/controller first; then use "
            "--clear left, --clear right, or Dobot's control software."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
