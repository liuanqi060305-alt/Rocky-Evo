import pickle
import unittest
from unittest import mock

import zmq

from dobot_control.robots.robot_node import (
    RobotServerError,
    ZMQClientRobot,
    _REMOTE_ERROR_KEY,
)


class ZMQClientRobotTests(unittest.TestCase):
    def _make_client(self, sockets):
        context = mock.Mock()
        context.socket.side_effect = sockets
        patcher = mock.patch(
            "dobot_control.robots.robot_node.zmq.Context",
            return_value=context,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        client = ZMQClientRobot(port=6002, timeout_ms=25)
        self.addCleanup(client.close)
        return client, context

    def test_timeout_reconnects_req_socket_for_the_next_request(self):
        timed_out = mock.Mock()
        recovered = mock.Mock()
        timed_out.recv.side_effect = zmq.Again()
        recovered.recv.return_value = pickle.dumps(14)
        client, context = self._make_client([timed_out, recovered])

        with self.assertRaisesRegex(TimeoutError, "Dobot controller alarm"):
            client.num_dofs()

        timed_out.close.assert_called_once_with(linger=0)
        self.assertEqual(client.num_dofs(), 14)
        self.assertEqual(context.socket.call_count, 2)

    def test_efsm_reconnects_and_reports_a_clear_error(self):
        invalid = mock.Mock()
        recovered = mock.Mock()
        invalid.send.side_effect = zmq.ZMQError(zmq.EFSM)
        recovered.recv.return_value = pickle.dumps(14)
        client, _ = self._make_client([invalid, recovered])

        with self.assertRaisesRegex(ConnectionError, "has been reconnected"):
            client.num_dofs()

        self.assertEqual(client.num_dofs(), 14)

    def test_remote_robot_exception_does_not_poison_req_state(self):
        socket = mock.Mock()
        socket.recv.side_effect = [
            pickle.dumps({
                _REMOTE_ERROR_KEY: {
                    "type": "AssertionError",
                    "message": "right robot error!",
                    "method": "get_observations",
                }
            }),
            pickle.dumps(14),
        ]
        client, context = self._make_client([socket])

        with self.assertRaisesRegex(RobotServerError, "right robot error"):
            client.get_observations()

        self.assertEqual(client.num_dofs(), 14)
        self.assertEqual(context.socket.call_count, 1)


if __name__ == "__main__":
    unittest.main()
