"""Offline checks for the Dobot 30003 motion socket request/reply contract."""

import socket
import time
import unittest
from unittest import mock

import numpy as np

from dobot_control.robots.dobot_api import DobotApiMove
from dobot_control.robots.dobot import DobotRobot


COMMAND = (
    "ServoJ(1.000000,2.000000,3.000000,4.000000,5.000000,"
    "6.000000,0.030000,gain=500)"
)
REPLY = "0,{}," + COMMAND + ";"


class FakeMotionSocket:
    """A scripted TCP byte stream; it never opens a network connection."""

    def __init__(self, recv_chunks=(), send_error=None):
        self.recv_chunks = list(recv_chunks)
        self.send_error = send_error
        self.sent = []
        self.connections = []
        self.timeout = None
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, address):
        self.connections.append(address)

    def sendall(self, data):
        self.sent.append(data)
        if self.send_error is not None:
            raise self.send_error

    def send(self, _data):
        raise AssertionError("motion commands must use sendall, not partial send")

    def recv(self, _size):
        if not self.recv_chunks:
            raise AssertionError("unexpected extra recv on fake motion socket")
        item = self.recv_chunks.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed = True


class MotionPortProtocolTests(unittest.TestCase):
    def make_api(self, fake):
        with mock.patch(
            "dobot_control.robots.dobot_api.socket.socket", return_value=fake
        ):
            api = DobotApiMove("127.0.0.1", 30003)
        self.addCleanup(api.close)
        return api

    @staticmethod
    def servo(api):
        return api.ServoJ(1, 2, 3, 4, 5, 6, 0.03)

    def test_servo_consumes_a_fragmented_reply(self):
        fake = FakeMotionSocket([
            b"0,{},ServoJ(1.000000,2.000000,",
            b"3.000000,4.000000,5.000000,6.000000,0.030000,gain=500);\r\n",
        ])
        api = self.make_api(fake)

        self.assertEqual(self.servo(api).strip(), REPLY)
        self.assertEqual(fake.sent, [COMMAND.encode("utf-8")])
        self.assertEqual(fake.connections, [("127.0.0.1", 30003)])

    def test_coalesced_extra_reply_is_not_misattributed_to_next_command(self):
        # This synchronous channel has exactly one outstanding command. A
        # second complete reply in the same recv is stale/unsolicited data.
        fake = FakeMotionSocket([REPLY.encode() + REPLY.encode()])
        api = self.make_api(fake)

        with self.assertRaises((RuntimeError, OSError)):
            self.servo(api)
        self.assertEqual(fake.sent, [COMMAND.encode("utf-8")])
        self.assertEqual(len(fake.connections), 1)

    def test_reply_line_ending_does_not_poison_the_next_command(self):
        fake = FakeMotionSocket([REPLY.encode() + b"\r\n", REPLY.encode()])
        api = self.make_api(fake)

        self.assertEqual(self.servo(api), REPLY)
        self.assertEqual(self.servo(api), REPLY)
        self.assertEqual(fake.sent, [COMMAND.encode("utf-8")] * 2)

    def test_nonzero_controller_reply_is_an_error(self):
        fake = FakeMotionSocket([f"-1,{{}},{COMMAND};".encode()])
        api = self.make_api(fake)

        with self.assertRaises((RuntimeError, OSError)):
            self.servo(api)
        self.assertEqual(fake.sent, [COMMAND.encode("utf-8")])

    def test_eof_during_reply_is_an_error(self):
        fake = FakeMotionSocket([b"0,{},ServoJ(", b""])
        api = self.make_api(fake)

        with self.assertRaises((RuntimeError, OSError)):
            self.servo(api)
        self.assertEqual(fake.sent, [COMMAND.encode("utf-8")])
        self.assertEqual(len(fake.connections), 1)

    def test_reply_timeout_is_not_swallowed_or_retried(self):
        fake = FakeMotionSocket([socket.timeout("simulated timeout")])
        api = self.make_api(fake)

        with self.assertRaises((RuntimeError, OSError)):
            self.servo(api)
        self.assertEqual(fake.sent, [COMMAND.encode("utf-8")])
        self.assertEqual(len(fake.connections), 1)

    def test_slow_fragments_cannot_extend_ack_deadline(self):
        class SlowSocket(FakeMotionSocket):
            def recv(self, size):
                time.sleep(0.03)
                return super().recv(size)

        fake = SlowSocket([REPLY[:12].encode(), REPLY[12:].encode()])
        api = self.make_api(fake)
        with self.assertRaises(TimeoutError):
            self.servo(api)
        self.assertEqual(fake.sent, [COMMAND.encode("utf-8")])

    def test_uncertain_ack_latches_channel_without_a_second_motion(self):
        fake = FakeMotionSocket([
            socket.timeout("simulated timeout"), REPLY.encode()
        ])
        api = self.make_api(fake)

        with self.assertRaises((RuntimeError, OSError)):
            self.servo(api)
        with self.assertRaises((RuntimeError, OSError)):
            self.servo(api)
        self.assertEqual(fake.sent, [COMMAND.encode("utf-8")])
        self.assertEqual(len(fake.connections), 1)

    def test_send_failure_never_reconnects_and_replays_motion(self):
        fake = FakeMotionSocket(send_error=BrokenPipeError("simulated send failure"))
        api = self.make_api(fake)

        with self.assertRaises((RuntimeError, OSError)):
            self.servo(api)
        self.assertEqual(fake.sent, [COMMAND.encode("utf-8")])
        self.assertEqual(len(fake.connections), 1)

    def test_motion_reply_failure_latches_robot_alarm(self):
        class BrokenMove:
            def ServoJ(self, *args):
                raise TimeoutError("motion acknowledgement timed out")

        robot = DobotRobot.__new__(DobotRobot)
        robot.robot = BrokenMove()
        robot.robot_ip = "test"
        robot.robot_is_err = False
        robot._use_gripper = False
        with self.assertRaises(TimeoutError):
            robot.command_joint_state(np.zeros(7, dtype=np.float64))
        self.assertTrue(robot.robot_is_err)


if __name__ == "__main__":
    unittest.main()
