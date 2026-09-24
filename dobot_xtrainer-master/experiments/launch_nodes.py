import sys
import os
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


def launch_robot_server(args: Args):
    port = args.robot_port
    # Query both controllers before enabling either arm or opening a gripper.
    # This avoids half-initializing the bimanual stack when one arm is already
    # in ERROR/COLLISION mode.
    statuses = query_all_controllers()
    for status in statuses:
        print(f"[PREFLIGHT] {format_status(status)}", flush=True)
    alarmed = [status for status in statuses if status.alarmed]
    if alarmed:
        names = ", ".join(status.name for status in alarmed)
        raise RuntimeError(
            f"Dobot controller alarm on {names}. Inspect the physical cell, "
            "then run 'python scripts/check_dobot_alarm.py --clear <side>' "
            "or clear it in Dobot control software before restarting launch_nodes.py."
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
