import unittest

import numpy as np

from experiments.run_inference_openpi import (
    classify_joint_increment,
    classify_workspace_position,
    select_key_input,
)


class InferenceKeySelectionTests(unittest.TestCase):
    def test_terminal_y_is_not_overwritten_by_opencv_residual_key(self):
        self.assertEqual(
            select_key_input("y", ord("f"), accepted_keys={"y", "f"}),
            "y",
        )

    def test_invalid_terminal_key_does_not_hide_valid_opencv_y(self):
        self.assertEqual(
            select_key_input("x", ord("y"), accepted_keys={"y", "f"}),
            "y",
        )

    def test_unlisted_safety_key_is_ignored(self):
        self.assertIsNone(
            select_key_input("r", 255, accepted_keys={"y", "f"})
        )

    def test_xyz_override_key_still_works(self):
        self.assertEqual(
            select_key_input(None, ord("I"), accepted_keys={"i", "f"}),
            "i",
        )

    def test_invalid_terminal_key_does_not_hide_window_control_key(self):
        self.assertEqual(
            select_key_input("x", ord("r"), accepted_keys={"r", "q"}),
            "r",
        )

    def test_opencv_no_key_value_is_ignored(self):
        self.assertIsNone(
            select_key_input(None, -1, accepted_keys={"y", "f"})
        )


class AutomaticSafetyDecisionTests(unittest.TestCase):
    @staticmethod
    def arm_delta(degrees):
        delta = np.zeros(12, dtype=np.float64)
        delta[3] = np.deg2rad(degrees)
        return delta

    @staticmethod
    def workspace(right_z=100.0):
        # [left XYZ/RPY, right XYZ/RPY]
        return np.array([
            0.0, -400.0, 100.0, 0.0, 0.0, 0.0,
            0.0, -400.0, right_z, 0.0, 0.0, 0.0,
        ])

    def test_small_joint_step_remains_unchanged(self):
        decision, maximum = classify_joint_increment(self.arm_delta(9.5))
        self.assertEqual(decision, "normal")
        self.assertAlmostEqual(maximum, 9.5)

    def test_recorded_size_joint_step_uses_existing_smoothing(self):
        decision, maximum = classify_joint_increment(self.arm_delta(21.97))
        self.assertEqual(decision, "smooth")
        self.assertAlmostEqual(maximum, 21.97)

    def test_unobserved_large_joint_step_is_rejected(self):
        decision, maximum = classify_joint_increment(self.arm_delta(25.01))
        self.assertEqual(decision, "reject")
        self.assertAlmostEqual(maximum, 25.01)

    def test_normal_workspace_is_safe(self):
        self.assertEqual(
            classify_workspace_position(self.workspace()),
            ("safe", []),
        )

    def test_only_right_low_z_is_the_known_insertion_zone(self):
        self.assertEqual(
            classify_workspace_position(self.workspace(right_z=35.0)),
            ("right_insertion_zone", ["right_z"]),
        )

    def test_right_z_below_task_floor_is_hard_limit(self):
        self.assertEqual(
            classify_workspace_position(self.workspace(right_z=29.9)),
            ("hard_limit", ["right_z"]),
        )

    def test_other_violation_is_not_hidden_by_insertion_zone(self):
        position = self.workspace(right_z=35.0)
        position[6] = 450.0
        self.assertEqual(
            classify_workspace_position(position),
            ("hard_limit", ["right_x", "right_z"]),
        )

    def test_left_low_z_remains_hard_limit(self):
        position = self.workspace(right_z=35.0)
        position[2] = 35.0
        self.assertEqual(
            classify_workspace_position(position),
            ("hard_limit", ["left_z", "right_z"]),
        )


if __name__ == "__main__":
    unittest.main()
