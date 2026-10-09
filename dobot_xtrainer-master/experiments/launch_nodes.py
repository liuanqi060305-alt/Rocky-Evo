import sys
import os
import ast
import re
import socket
# 获取根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from dataclasses import dataclass
import tyro
from dobot_control.robots.dobot import DobotRobot
from dobot_control.robots.robot import BimanualRobot, PrintRobot
from dobot_control.robots.robot_node import ZMQServerRobot
from scripts.check_dobot_alarm import format_status, query_all_controllers


@dataclass
class Args:
    robot_port: int = 6002
    hostname: str = "127.0.0.1"


def ensure_robot_server_port_free(host: str, port: int) -> None:
    """Reject a second robot server before connecting to either controller."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    except OSError as exc:
        raise RuntimeError(
            f"Robot server port {host}:{port} is unavailable; an existing "
            "launch_nodes.py may still be running. Inspect that process before "
            "starting another robot server."
        ) from exc


def controller_ready_for_launch(status) -> bool:
    """Allow only clear, parseable DISABLED or ENABLED_IDLE controller states."""
    mode = re.fullmatch(
        r"\s*0,\{\s*([45])\s*\},RobotMode\(\);?\s*",
        status.mode_response or "",
    )
    if mode is None or int(mode.group(1)) != status.mode or status.error_ids:
        return False
    errors = re.fullmatch(
        r"\s*0,\{(.+)\},GetErrorID\(\);?\s*",
        status.error_response or "",
    )
    if errors is None:
        return False
    try:
        payload = ast.literal_eval(errors.group(1))
    except (SyntaxError, ValueError):
        return False

    def contains_only_empty_or_zero(value):
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return value == 0
        if isinstance(value, (list, tuple)):
            return all(contains_only_empty_or_zero(item) for item in value)
        return False

    return isinstance(payload, (list, tuple)) and contains_only_empty_or_zero(payload)


def launch_robot_server(args: Args):
    port = args.robot_port
    ensure_robot_server_port_free(args.hostname, port)
    # Query both controllers before enabling either arm or opening a gripper.
    # This avoids half-initializing the bimanual stack when one arm is already
    # in ERROR/COLLISION mode.
    statuses = query_all_controllers()
    for status in statuses:
        print(f"[PREFLIGHT] {format_status(status)}", flush=True)
    unready = [status for status in statuses if not controller_ready_for_launch(status)]
    if len(statuses) != 2 or {status.name for status in statuses} != {"left", "right"} or unready:
        names = ", ".join(status.name for status in unready) or "left/right"
        raise RuntimeError(
            f"Dobot controller preflight is not clear on {names}. Inspect the "
            "physical cell and controller alarms before restarting launch_nodes.py."
        )
    _robot_l = DobotRobot(robot_ip="192.168.5.1", robot_number=2)  # IP of the left hand robotic arm
    _robot_r = DobotRobot(robot_ip="192.168.5.2", robot_number=2)  # IP of the rigth hand robotic arm
    robot = BimanualRobot(_robot_l, _robot_r)
    server = ZMQServerRobot(robot, port=port, host=args.hostname)
    print(f"Starting robot server on port {port}")
    server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
