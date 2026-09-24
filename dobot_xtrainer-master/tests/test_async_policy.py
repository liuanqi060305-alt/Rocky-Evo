import time
import unittest

import numpy as np

from dobot_control.agents.async_policy import (
    AsyncPolicyRunner,
    TemporalActionEnsembler,
    prepare_action_chunk,
)


class FakePolicy:
    def __init__(self, delay=0.02):
        self.delay = delay
        self.calls = 0

    def infer(self, observation):
        self.calls += 1
        time.sleep(self.delay)
        return {"actions": np.full((50, 14), observation["value"], dtype=np.float64)}


class AsyncPolicyRunnerTests(unittest.TestCase):
    def test_single_flight_and_result_metadata(self):
        policy = FakePolicy()
        runner = AsyncPolicyRunner(policy)
        self.addCleanup(runner.close)
        request_id = runner.submit({"value": 3.0}, generation=7, request_step=11)
        self.assertIsNotNone(request_id)
        self.assertIsNone(runner.submit({"value": 4.0}, generation=7, request_step=12))
        result = runner.wait(request_id, timeout_s=1.0)
        self.assertIsNone(result.error)
        self.assertEqual(result.generation, 7)
        self.assertEqual(result.request_step, 11)
        self.assertTrue(np.all(result.output["actions"] == 3.0))
        self.assertTrue(runner.wait_until_idle(1.0))

    def test_generation_is_preserved_for_late_result_rejection(self):
        runner = AsyncPolicyRunner(FakePolicy())
        self.addCleanup(runner.close)
        request_id = runner.submit({"value": 1.0}, generation=2, request_step=0)
        result = runner.wait(request_id, timeout_s=1.0)
        self.assertEqual(result.generation, 2)

    def test_latency_skip_and_three_step_arm_blend(self):
        actions = np.ones((10, 14), dtype=np.float64)
        actions[:, 6] = 0.8
        actions[:, 13] = 0.6
        chunk, index = prepare_action_chunk(
            actions,
            steps_elapsed=4,
            last_action=np.zeros(14),
            blend_steps=3,
        )
        self.assertEqual(index, 4)
        np.testing.assert_allclose(chunk[4, :6], 1 / 3)
        np.testing.assert_allclose(chunk[5, :6], 2 / 3)
        np.testing.assert_allclose(chunk[6, :6], 1.0)
        self.assertEqual(chunk[4, 6], 0.8)
        self.assertEqual(chunk[4, 13], 0.6)

    def test_latency_skip_is_clamped_to_last_action(self):
        chunk, index = prepare_action_chunk(
            np.zeros((4, 14)),
            steps_elapsed=99,
            last_action=np.zeros(14),
            blend_steps=0,
        )
        self.assertEqual(index, 3)
        self.assertEqual(chunk.shape, (4, 14))

    def test_temporal_ensemble_aligns_chunks_by_rollout_step(self):
        ensemble = TemporalActionEnsembler(max_chunks=3, decay=0.7)
        old = np.zeros((20, 14), dtype=np.float64)
        old[:, 6] = 0.1
        old[:, 13] = 0.2
        new = np.full((20, 14), 2.0, dtype=np.float64)
        new[:, 6] = 0.9
        new[:, 13] = 0.8
        ensemble.add(old, request_step=0)
        ensemble.add(new, request_step=5)

        action, newest_index, sources = ensemble.action_for_step(8)

        newest_weight = 1.0 / (1.0 + np.exp(-0.7))
        np.testing.assert_allclose(action[:6], 2.0 * newest_weight)
        np.testing.assert_allclose(action[7:13], 2.0 * newest_weight)
        self.assertEqual(action[6], 0.9)
        self.assertEqual(action[13], 0.8)
        self.assertEqual(newest_index, 3)
        self.assertEqual(sources, 2)

    def test_temporal_ensemble_expires_and_clear_removes_old_rollout(self):
        ensemble = TemporalActionEnsembler(max_chunks=2)
        ensemble.add(np.zeros((4, 14)), request_step=3)
        with self.assertRaises(IndexError):
            ensemble.action_for_step(7)
        ensemble.add(np.zeros((4, 14)), request_step=10)
        ensemble.clear()
        with self.assertRaises(IndexError):
            ensemble.action_for_step(10)


if __name__ == "__main__":
    unittest.main()
