import unittest

import numpy as np

from experiments.run_inference_openpi import (
    LEFT_PICKUP_MAX_DURATION_S,
    classify_joint_increment,
    classify_left_action_target,
    classify_left_target_position,
    classify_workspace_position,
    left_pickup_zone_timed_out,
    nova2_tcp_position_mm,
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
    def workspace(left_x=0.0, left_y=-400.0, left_z=100.0,
                  right_z=100.0):
        # [left XYZ/RPY, right XYZ/RPY]
        return np.array([
            left_x, left_y, left_z, 0.0, 0.0, 0.0,
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

    def test_only_left_low_z_inside_local_box_is_pickup_zone(self):
        self.assertEqual(
            classify_workspace_position(self.workspace(
                left_x=-173.0, left_y=-390.0, left_z=35.0,
            )),
            ("left_pickup_zone", ["left_z"]),
        )

    def test_left_z_below_task_floor_is_hard_limit(self):
        self.assertEqual(
            classify_workspace_position(self.workspace(
                left_x=-173.0, left_y=-390.0, left_z=29.9,
            )),
            ("hard_limit", ["left_z"]),
        )

    def test_left_low_z_outside_local_box_is_hard_limit(self):
        self.assertEqual(
            classify_workspace_position(self.workspace(
                left_x=-157.0, left_y=-390.0, left_z=35.0,
            )),
            ("hard_limit", ["left_z"]),
        )

    def test_left_pickup_zone_does_not_hide_other_violation(self):
        position = self.workspace(
            left_x=-173.0, left_y=-390.0, left_z=35.0,
        )
        position[0] = 350.0
        self.assertEqual(
            classify_workspace_position(position),
            ("hard_limit", ["left_x", "left_z"]),
        )

    def test_other_violation_is_not_hidden_by_insertion_zone(self):
        position = self.workspace(right_z=35.0)
        position[6] = 450.0
        self.assertEqual(
            classify_workspace_position(position),
            ("hard_limit", ["right_x", "right_z"]),
        )

    def test_both_arms_low_z_remain_hard_limit(self):
        position = self.workspace(
            left_x=-173.0, left_y=-390.0, left_z=35.0, right_z=35.0,
        )
        self.assertEqual(
            classify_workspace_position(position),
            ("hard_limit", ["left_z", "right_z"]),
        )

    def test_two_recorded_l2_targets_are_inside_left_pickup_zone(self):
        recorded_actions = (
            np.array([
                -1.5041656675772828, -0.4000949056704668,
                -2.018889698471504, 0.8529576665924457,
                1.2802766983679823, 1.571628148668516,
                0.15632341420836784, 1.5202014748960926,
                -0.21199467918958392, 1.7551993035397722,
                0.11164312802515679, -1.372415034996108,
                -1.792451766685299, 0.9742149649444061,
            ]),
            np.array([
                -1.497616508141097, -0.40863882284239045,
                -2.017157079349956, 0.8577936090456846,
                1.273691074661955, 1.6284031849349343,
                0.32052748082002264, 1.5234912939403593,
                -0.20351883751692737, 1.7426167455812847,
                0.11885516558709931, -1.3732291985310239,
                -1.8000547042560018, 0.9744439104261277,
            ]),
        )
        expected_targets = (
            np.array([-173.451086, -391.582303, 40.036259]),
            np.array([-172.637487, -393.346224, 38.389905]),
        )
        for action, expected in zip(recorded_actions, expected_targets):
            classification, target = classify_left_action_target(action)
            self.assertEqual(classification, "left_pickup_zone")
            np.testing.assert_allclose(target, expected, atol=1e-6)
            np.testing.assert_allclose(
                nova2_tcp_position_mm(action[:6]), expected, atol=1e-6
            )

    def test_pending_left_target_below_floor_is_hard_limit(self):
        action = np.array([
            -1.497616508141097, -0.40863882284239045,
            -2.067157079349956, 0.8577936090456846,
            1.273691074661955, 1.6284031849349343,
        ])
        classification, target = classify_left_action_target(action)
        self.assertEqual(classification, "hard_limit")
        self.assertLessEqual(target[2], 30.0)

    def test_high_left_target_does_not_add_approximate_fk_xy_limit(self):
        self.assertEqual(
            classify_left_target_position([1000.0, 1000.0, 100.0]),
            "safe",
        )

    def test_low_left_target_position_must_stay_inside_pickup_box(self):
        self.assertEqual(
            classify_left_target_position([-173.0, -390.0, 29.9]),
            "hard_limit",
        )
        self.assertEqual(
            classify_left_target_position([-157.0, -390.0, 35.0]),
            "hard_limit",
        )
        self.assertEqual(
            classify_left_target_position([-173.0, -390.0, 35.0]),
            "left_pickup_zone",
        )

    def test_malformed_left_action_target_fails_closed(self):
        for action in ([0.0] * 5, ["bad"] * 6):
            classification, target = classify_left_action_target(action)
            self.assertEqual(classification, "hard_limit")
            self.assertTrue(np.isnan(target).all())

    def test_left_pickup_zone_timeout_is_strictly_over_two_seconds(self):
        entered_at = 10.0
        self.assertFalse(left_pickup_zone_timed_out(
            entered_at, entered_at + LEFT_PICKUP_MAX_DURATION_S,
        ))
        self.assertTrue(left_pickup_zone_timed_out(
            entered_at, entered_at + LEFT_PICKUP_MAX_DURATION_S + 0.001,
        ))


if __name__ == "__main__":
    unittest.main()
