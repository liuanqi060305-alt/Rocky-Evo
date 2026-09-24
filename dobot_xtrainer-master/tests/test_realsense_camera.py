"""Camera lifecycle regressions; no camera or robot hardware is accessed."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from dobot_control.cameras import realsense_camera as camera


class RealSenseCameraTests(unittest.TestCase):
    def setUp(self):
        self.frame = Mock()
        self.frame.get_data.return_value = np.zeros((4, 6, 3), dtype=np.uint8)
        self.frames = Mock()
        self.frames.get_color_frame.return_value = self.frame
        self.pipeline = Mock()
        self.pipeline.wait_for_frames.return_value = self.frames
        self.device = Mock()
        self.device.get_info.return_value = "test"
        self.context = Mock()
        self.context.query_devices.return_value = [self.device]
        for target, kwargs in (
            ("rs.pipeline", {"return_value": self.pipeline}),
            ("rs.config", {}),
            ("rs.context", {"return_value": self.context}),
            ("time.sleep", {}),
        ):
            p = patch.object(camera.rs if target.startswith("rs.") else camera.time,
                             target.split(".")[1], **kwargs)
            mocked = p.start()
            self.addCleanup(p.stop)
            if target == "rs.pipeline":
                self.factory = mocked

    def test_device_enumeration_does_not_reset_cameras(self):
        self.assertEqual(camera.get_device_ids(), ["test"])
        self.device.hardware_reset.assert_not_called()

    def test_slow_first_frame_then_short_runtime_timeout(self):
        def wait(timeout):
            # Simulate a camera requiring 700ms to produce its first frame.
            if self.pipeline.wait_for_frames.call_count == 1 and timeout < 700:
                raise RuntimeError("startup interrupted too early")
            return self.frames

        self.pipeline.wait_for_frames.side_effect = wait
        cam = camera.RealSenseCamera("test")
        self.addCleanup(cam.close)
        self.factory.assert_called_once()
        calls = self.pipeline.wait_for_frames.call_args_list
        self.assertGreaterEqual(calls[0].args[0], 700)
        self.assertTrue(all(c.args == (camera.FRAME_TIMEOUT_MS,) for c in calls[1:]))
        image, depth = cam.read()
        self.assertEqual(image.shape, (4, 6, 3))
        self.assertEqual(depth.shape, (4, 6, 1))

    def test_restart_allows_slow_first_frame_again(self):
        cam = camera.RealSenseCamera("test")
        self.addCleanup(cam.close)
        self.pipeline.wait_for_frames.reset_mock()
        self.pipeline.wait_for_frames.side_effect = (
            [RuntimeError("lost stream")] * camera.FRAME_WAITS_BEFORE_RESTART
            + [self.frames]
        )
        cam.read()
        self.assertEqual([c.args[0] for c in self.pipeline.wait_for_frames.call_args_list],
                         [camera.FRAME_TIMEOUT_MS] * camera.FRAME_WAITS_BEFORE_RESTART
                         + [camera.STARTUP_TIMEOUT_MS])
        self.pipeline.stop.assert_called_once()

    def test_transient_timeout_does_not_restart_pipeline(self):
        cam = camera.RealSenseCamera("test")
        self.addCleanup(cam.close)
        self.pipeline.stop.reset_mock()
        self.pipeline.wait_for_frames.side_effect = [RuntimeError("brief stall"), self.frames]
        cam.read()
        self.pipeline.stop.assert_not_called()
        self.device.hardware_reset.assert_not_called()

    def test_repeated_rebuild_failures_trigger_one_device_hardware_reset(self):
        # Construction is still waiting for a first frame, so each pipeline
        # gets one long startup wait before the fourth failure permits reset.
        failures = [RuntimeError("uvc stalled")] * camera.HARD_RESET_AFTER_FAILURES
        self.pipeline.wait_for_frames.side_effect = failures + [self.frames] * 50
        cam = camera.RealSenseCamera("test")
        self.addCleanup(cam.close)
        self.device.hardware_reset.assert_called_once_with()
        self.assertEqual(self.factory.call_count, camera.HARD_RESET_AFTER_FAILURES + 1)

    def test_hardware_reset_can_be_disabled_for_live_inference(self):
        cam = camera.RealSenseCamera("test", allow_hardware_reset=False)
        self.addCleanup(cam.close)
        self.device.hardware_reset.reset_mock()
        self.pipeline.wait_for_frames.side_effect = RuntimeError("unstable usb")
        with self.assertRaisesRegex(RuntimeError, "unrecoverable"):
            cam.read()
        self.device.hardware_reset.assert_not_called()

    def test_manual_exposure_and_gain_are_applied_after_stream_start(self):
        profile = Mock()
        color_sensor = Mock()
        stream_profile = Mock()
        stream_profile.stream_type.return_value = camera.rs.stream.color
        color_sensor.get_stream_profiles.return_value = [stream_profile]
        profile.get_device.return_value.query_sensors.return_value = [color_sensor]
        color_sensor.supports.return_value = True
        color_sensor.get_option_range.return_value = SimpleNamespace(
            min=1.0, max=165000.0
        )
        color_sensor.get_option.side_effect = lambda option: (
            11000.0 if option == camera.rs.option.exposure else 16.0
        )
        self.pipeline.start.return_value = profile

        cam = camera.RealSenseCamera("test", exposure_us=11000, gain=16)
        self.addCleanup(cam.close)

        self.assertEqual(
            color_sensor.set_option.call_args_list,
            [
                call(camera.rs.option.enable_auto_exposure, 0.0),
                call(camera.rs.option.exposure, 11000.0),
                call(camera.rs.option.gain, 16.0),
            ],
        )

    def test_exhausted_retries_release_every_pipeline_without_extra_restart(self):
        self.pipeline.wait_for_frames.side_effect = RuntimeError("no frames")
        with self.assertRaisesRegex(RuntimeError, "unrecoverable"):
            camera.RealSenseCamera("test")
        self.assertEqual(self.factory.call_count, camera.READ_RETRIES)
        self.assertEqual(self.pipeline.stop.call_count, camera.READ_RETRIES)

    def test_empty_color_frames_raise_read_error(self):
        empty_frame = Mock()
        empty_frame.__bool__ = Mock(return_value=False)
        self.frames.get_color_frame.return_value = empty_frame
        with self.assertRaisesRegex(RuntimeError, "no color frame"):
            camera.RealSenseCamera("test")
        empty_frame.get_data.assert_not_called()

    def test_start_failure_releases_pipeline(self):
        self.pipeline.start.side_effect = RuntimeError("start failed")
        with patch.object(camera.RealSenseCamera, "_diagnose_start_failure", return_value="start failed"):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                camera.RealSenseCamera("test")
        self.pipeline.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
