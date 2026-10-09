import unittest
from threading import Event, Lock
from unittest import mock

from dobot_control.robots.dobot import DobotRobot
from dobot_control.robots.dobot_api import DobotApiStatus
from dobot_control.robots.robot import BimanualRobot

from scripts.check_dobot_alarm import (
    ControllerStatus,
    format_status,
    parse_error_ids,
    parse_robot_mode,
)


class DobotAlarmParsingTests(unittest.TestCase):
    def test_parse_modes(self):
        self.assertEqual(parse_robot_mode("0,{9},RobotMode();"), 9)
        self.assertEqual(parse_robot_mode("0,{5},RobotMode();"), 5)
        self.assertIsNone(parse_robot_mode("bad response"))

    def test_parse_flat_and_legacy_nested_error_ids(self):
        self.assertEqual(
            parse_error_ids("0,{[1537,2048]},GetErrorID();"),
            [1537, 2048],
        )
        self.assertEqual(
            parse_error_ids("0,{[[-2],[],[],[],[],[],[]]},GetErrorID();"),
            [-2],
        )

    def test_status_marks_only_error_and_collision_modes_as_alarm(self):
        normal = ControllerStatus("right", "ip", 5, [], "", "")
        alarm = ControllerStatus("left", "ip", 9, [-2], "", "")
        self.assertFalse(normal.alarmed)
        self.assertTrue(alarm.alarmed)
        self.assertIn("generic", format_status(alarm))


class DobotFeedbackFailureTests(unittest.TestCase):
    def test_bimanual_status_reads_both_latched_arms(self):
        left = mock.Mock(robot_is_err=True)
        right = mock.Mock(robot_is_err=True)
        left.get_controller_status.return_value = {"mode_response": "left"}
        right.get_controller_status.return_value = {"mode_response": "right"}
        robot = BimanualRobot.__new__(BimanualRobot)
        robot._robot_l = left
        robot._robot_r = right

        self.assertEqual(robot.get_controller_statuses(), {
            "left": {"mode_response": "left"},
            "right": {"mode_response": "right"},
        })
        left.get_controller_status.assert_called_once_with()
        right.get_controller_status.assert_called_once_with()

    def test_controller_status_remains_readable_when_motion_is_latched(self):
        class FakeDashboard:
            def RobotMode(self):
                return "0,{9},RobotMode();"

            def GetErrorID(self):
                return "0,{[116]},GetErrorID();"

        robot = DobotRobot.__new__(DobotRobot)
        robot.robot_is_err = True
        robot.r_inter = FakeDashboard()
        self.assertEqual(robot.get_controller_status(), {
            "mode_response": "0,{9},RobotMode();",
            "error_response": "0,{[116]},GetErrorID();",
        })

    def test_closed_status_socket_raises_instead_of_spinning(self):
        class ClosedSocket:
            def recv(self, count):
                return b""

            def close(self):
                pass

        status = DobotApiStatus.__new__(DobotApiStatus)
        status.ip = "test"
        status.port = 30004
        status.socket_dobot = ClosedSocket()
        with self.assertRaises(ConnectionError):
            status.get_error()

    def test_status_thread_latches_error_on_feedback_failure(self):
        class BrokenStatus:
            def get_error(self):
                raise TimeoutError("status timeout")

        robot = DobotRobot.__new__(DobotRobot)
        robot.robot_ip = "test"
        robot.robot_status = BrokenStatus()
        robot.robot_is_err = False
        robot._stop_thread = Event()
        robot._lock = Lock()
        robot.get_robot_err()
        self.assertTrue(robot.robot_is_err)


if __name__ == "__main__":
    unittest.main()
