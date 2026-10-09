"""Inference fault handling tests; these tests never connect to robot hardware."""

import csv
import json
import pathlib
import tempfile
import unittest

import numpy as np

from dobot_control.robots.robot_node import RobotServerError
from scripts.check_dobot_alarm import ControllerStatus
from experiments.run_inference_openpi import (
    Args,
    JointTrackingGuard,
    RolloutRecorder,
    controllers_ready_for_recovery,
    finalize_recorded_rollout,
    read_controller_statuses,
    finish_robot_fault,
    is_controller_alarm,
)


def right_arm_pose(j3_deg, j4_deg=-65.0):
    pose = np.zeros(14, dtype=np.float64)
    pose[9] = np.deg2rad(j3_deg)
    pose[10] = np.deg2rad(j4_deg)
    return pose


class ControllerAlarmClassificationTests(unittest.TestCase):
    def test_real_robot_server_alarm_is_recognized(self):
        for arm, method in (
            ("right", "command_joint_state"),
            ("left", "get_observations"),
        ):
            with self.subTest(arm=arm, method=method):
                error = RobotServerError(
                    f"robot server '{method}' failed with "
                    f"AssertionError: {arm} robot error!"
                )
                self.assertTrue(is_controller_alarm(error))

    def test_unrelated_server_error_is_not_called_a_controller_alarm(self):
        error = RobotServerError(
            "robot server 'command_joint_state' failed with "
            "ValueError: invalid action shape"
        )
        self.assertFalse(is_controller_alarm(error))

    def test_recovery_requires_both_arms_idle_and_alarm_free(self):
        def status(name, mode=5, errors=None,
                   reply="0,{[[],[],[],[],[],[],[]]},GetErrorID();"):
            return ControllerStatus(
                name=name, ip="127.0.0.1", mode=mode,
                error_ids=[] if errors is None else errors,
                mode_response=f"0,{{{mode}}},RobotMode();",
                error_response=reply,
            )

        self.assertTrue(controllers_ready_for_recovery(
            [status("left"), status("right")]
        ))
        self.assertFalse(controllers_ready_for_recovery(
            [status("left"), status("right", mode=9, errors=[116])]
        ))
        self.assertFalse(controllers_ready_for_recovery(
            [status("left"), status("right", mode=4)]
        ))
        self.assertFalse(controllers_ready_for_recovery(
            [status("left"), status("right", reply="malformed")]
        ))
        self.assertFalse(controllers_ready_for_recovery(
            [status("left"), status("right", reply="0,{garbage},GetErrorID();")]
        ))
        self.assertFalse(controllers_ready_for_recovery(
            [status("left"), status("right", reply="0,{},GetErrorID();")]
        ))

    def test_recovery_status_uses_existing_robot_server_connection(self):
        class FakeClient:
            def get_controller_statuses(self):
                return {
                    name: {
                        "mode_response": "0,{5},RobotMode();",
                        "error_response": "0,{[[],[],[],[],[],[],[]]},GetErrorID();",
                    }
                    for name in ("left", "right")
                }

        self.assertTrue(controllers_ready_for_recovery(
            read_controller_statuses(FakeClient())
        ))


class JointTrackingGuardTests(unittest.TestCase):
    def test_insertion_pose_reversal_with_growing_lag_triggers(self):
        guard = JointTrackingGuard()
        self.assertFalse(guard.observe(
            right_arm_pose(70.0), right_arm_pose(70.0)
        )[0])
        self.assertFalse(guard.observe(
            right_arm_pose(70.8), right_arm_pose(70.2)
        )[0])

        # The commanded joint keeps advancing while the measured joint
        # reverses sharply. No real robot is commanded by this test.
        triggered, details = guard.observe(
            right_arm_pose(71.2), right_arm_pose(69.4)
        )
        self.assertTrue(triggered)
        self.assertTrue(details)

    def test_large_lag_while_joint_still_moves_forward_is_not_contact(self):
        guard = JointTrackingGuard()
        for target, actual in ((70.0, 67.5), (70.6, 68.1), (71.2, 68.7)):
            with self.subTest(target=target):
                triggered, _ = guard.observe(
                    right_arm_pose(target), right_arm_pose(actual)
                )
                self.assertFalse(triggered)

    def test_guard_reset_discards_previous_joint_history(self):
        guard = JointTrackingGuard()
        guard.observe(right_arm_pose(70.0), right_arm_pose(70.0))
        guard.observe(right_arm_pose(70.8), right_arm_pose(70.2))
        guard.reset()
        triggered, _ = guard.observe(
            right_arm_pose(71.2), right_arm_pose(69.4)
        )
        self.assertFalse(triggered)


class RolloutFailureAccountingTests(unittest.TestCase):
    def test_alarm_during_success_finalization_cannot_be_counted_as_success(self):
        class AlarmEnv:
            def get_obs(self):
                raise RobotServerError(
                    "robot server 'get_observations' failed with "
                    "AssertionError: right robot error!"
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = RolloutRecorder(
                Args(rollout_root=temp_dir, save_images=False),
                server_metadata={},
            )
            recorder.start()
            meta, fault = finalize_recorded_rollout(
                recorder, AlarmEnv(), "success", step=7,
            )
            self.assertIsInstance(fault, RobotServerError)
            self.assertEqual(meta["result"], "failure")
            self.assertEqual(recorder.summary["success"], 0)
            self.assertEqual(recorder.summary["failure"], 1)

    def test_failed_idle_light_holds_instead_of_recording_aborted(self):
        class BrokenLightEnv:
            def set_do_status(self, which_do):
                raise TimeoutError("robot server did not reply")

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = RolloutRecorder(
                Args(rollout_root=temp_dir, save_images=False),
                server_metadata={},
            )
            recorder.start()
            meta, fault = finalize_recorded_rollout(
                recorder, BrokenLightEnv(), "aborted", "operator_reset", step=7,
            )
            self.assertIsInstance(fault, TimeoutError)
            self.assertEqual(meta["result"], "failure")
            self.assertEqual(recorder.summary["failure"], 1)

    def test_controller_alarm_failure_updates_csv_and_success_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = Args(
                rollout_root=temp_dir,
                task_name="fixture_task",
                condition="original_position",
                save_images=False,
            )
            recorder = RolloutRecorder(args, server_metadata={})

            recorder.start()
            recorder.finish("success")
            recorder.start()
            recorder.record(
                state=np.zeros(14),
                action=np.zeros(14),
                infer_latency_ms=100,
                chunk_index=0,
                images=(),
            )
            alarm = RobotServerError(
                "robot server 'command_joint_state' failed with "
                "AssertionError: right robot error!"
            )
            failed_meta = finish_robot_fault(recorder, alarm, step=1)
            self.assertIsNone(finish_robot_fault(recorder, alarm, step=1))

            root = pathlib.Path(temp_dir)
            with (root / "rollout_results.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            with (root / "success_rate_summary.json").open(
                encoding="utf-8"
            ) as handle:
                summary = json.load(handle)

            self.assertEqual([row["result"] for row in rows], ["success", "failure"])
            self.assertIn("right robot error!", rows[1]["failure_reason"])
            self.assertEqual(rows[1]["num_steps"], "1")
            self.assertEqual(summary["success"], 1)
            self.assertEqual(summary["failure"], 1)
            self.assertEqual(summary["evaluated"], 2)
            self.assertEqual(summary["success_rate"], 0.5)
            self.assertEqual(failed_meta["events"][0]["type"], "robot_controller_alarm")
            self.assertEqual(failed_meta["events"][0]["kind"], "controller_alarm")


if __name__ == "__main__":
    unittest.main()
