"""Offline checks that startup rejects unsafe or ambiguous controller state."""

import unittest
from contextlib import ExitStack
from unittest import mock

from experiments import launch_nodes
from scripts.check_dobot_alarm import ControllerStatus


EMPTY_ERRORS = "0,{[[],[],[],[],[],[],[]]},GetErrorID();"


def status(
    name,
    *,
    mode=5,
    error_ids=None,
    mode_response=None,
    error_response=EMPTY_ERRORS,
):
    return ControllerStatus(
        name=name,
        ip={"left": "192.168.5.1", "right": "192.168.5.2"}[name],
        mode=mode,
        error_ids=[] if error_ids is None else error_ids,
        mode_response=(
            f"0,{{{mode}}},RobotMode();"
            if mode_response is None else mode_response
        ),
        error_response=error_response,
    )


class LaunchNodesPreflightTests(unittest.TestCase):
    def setUp(self):
        self.args = launch_nodes.Args(hostname="127.0.0.1", robot_port=6002)

    def test_port_probe_reports_bind_failure_without_real_socket(self):
        with mock.patch.object(launch_nodes.socket, "socket") as socket_factory:
            probe = socket_factory.return_value.__enter__.return_value
            probe.bind.side_effect = OSError("address already in use")
            with self.assertRaisesRegex(RuntimeError, "port.*6002"):
                launch_nodes.ensure_robot_server_port_free("127.0.0.1", 6002)

        probe.bind.assert_called_once_with(("127.0.0.1", 6002))

    def test_occupied_robot_server_port_aborts_before_controller_query(self):
        with ExitStack() as stack:
            check_port = stack.enter_context(mock.patch.object(
                launch_nodes,
                "ensure_robot_server_port_free",
                side_effect=RuntimeError("port 6002 is already occupied"),
            ))
            query = stack.enter_context(
                mock.patch.object(launch_nodes, "query_all_controllers"))
            robot = stack.enter_context(
                mock.patch.object(launch_nodes, "DobotRobot"))
            with self.assertRaisesRegex(RuntimeError, "port 6002"):
                launch_nodes.launch_robot_server(self.args)

        check_port.assert_called_once_with("127.0.0.1", 6002)
        query.assert_not_called()
        robot.assert_not_called()

    def test_bad_controller_status_aborts_before_any_robot_initializes(self):
        valid_left = status("left", mode=4)
        bad_right_cases = {
            "controller alarm": status("right", mode=9, error_ids=[116]),
            "collision": status("right", mode=11),
            "unknown mode": status("right", mode=12),
            "unparsed mode": status("right", mode=None, mode_response="garbage"),
            "nonempty errors in idle": status("right", mode=5, error_ids=[116],
                                              error_response="0,{[116]},GetErrorID();"),
            "malformed mode reply": status("right", mode=5,
                                            mode_response="malformed reply"),
            "malformed error reply": status("right", mode=5,
                                             error_response="malformed reply"),
            "failed mode query": status("right", mode=5,
                                         mode_response="-1,{5},RobotMode();"),
            "failed error query": status("right", mode=5,
                                          error_response="-1,{[]},GetErrorID();"),
        }
        for case, bad_right in bad_right_cases.items():
            with self.subTest(case=case):
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(
                        launch_nodes, "ensure_robot_server_port_free"))
                    stack.enter_context(mock.patch.object(
                        launch_nodes, "query_all_controllers",
                        return_value=[valid_left, bad_right]))
                    robot = stack.enter_context(mock.patch.object(
                        launch_nodes, "DobotRobot"))
                    bimanual = stack.enter_context(mock.patch.object(
                        launch_nodes, "BimanualRobot"))
                    server = stack.enter_context(mock.patch.object(
                        launch_nodes, "ZMQServerRobot"))
                    with self.assertRaises(RuntimeError):
                        launch_nodes.launch_robot_server(self.args)

                robot.assert_not_called()
                bimanual.assert_not_called()
                server.assert_not_called()

    def test_missing_controller_status_aborts_before_robot_initialization(self):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                launch_nodes, "ensure_robot_server_port_free"))
            stack.enter_context(mock.patch.object(
                launch_nodes, "query_all_controllers",
                return_value=[status("left")]))
            robot = stack.enter_context(mock.patch.object(
                launch_nodes, "DobotRobot"))
            with self.assertRaises(RuntimeError):
                launch_nodes.launch_robot_server(self.args)

        robot.assert_not_called()

    def test_clear_disabled_and_idle_controllers_can_launch(self):
        with ExitStack() as stack:
            check_port = stack.enter_context(mock.patch.object(
                launch_nodes, "ensure_robot_server_port_free"))
            stack.enter_context(mock.patch.object(
                launch_nodes, "query_all_controllers",
                return_value=[status("left", mode=4), status("right", mode=5)]))
            robot = stack.enter_context(mock.patch.object(
                launch_nodes, "DobotRobot"))
            bimanual = stack.enter_context(mock.patch.object(
                launch_nodes, "BimanualRobot"))
            server = stack.enter_context(mock.patch.object(
                launch_nodes, "ZMQServerRobot"))
            launch_nodes.launch_robot_server(self.args)

        check_port.assert_called_once_with("127.0.0.1", 6002)
        self.assertEqual(robot.call_count, 2)
        bimanual.assert_called_once()
        server.assert_called_once()
        server.return_value.serve.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
