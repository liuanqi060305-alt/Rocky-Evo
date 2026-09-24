import unittest

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


if __name__ == "__main__":
    unittest.main()
