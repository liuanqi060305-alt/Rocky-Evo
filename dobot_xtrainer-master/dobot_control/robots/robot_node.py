import pickle
import threading
from typing import Any, Dict

import numpy as np
import zmq
import time
from dobot_control.robots.robot import Robot
from scripts.function_util import log_write

DEFAULT_ROBOT_PORT = 6000

_REMOTE_ERROR_KEY = "__robot_server_error__"


class RobotServerError(RuntimeError):
    """The robot server received the request but its robot method failed."""


class ZMQServerRobot:
    def __init__(
        self,
        robot: Robot,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
    ):
        self.port = port
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Robot Sever Binding to {addr}, Robot: {robot}"
        print(debug_message)
        self._socket.bind(addr)
        self._stop_event = threading.Event()

        self.ON = 1
        self.OFF = 0
        self.Enable = self.OFF
        self.Follow = self.OFF
        self.Record = self.OFF

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        # Wake periodically so stop() can be observed without flooding stdout
        # while no client request is pending.
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        print("*"*100)
        while not self._stop_event.is_set():
            try:
                # Wait for next request from client
                message = self._socket.recv()
            except zmq.Again:
                continue

            # A REP socket must send exactly one reply after every successful
            # recv.  If a robot assertion escapes here, the server thread dies
            # and the client first times out, then leaves its REQ socket in the
            # EFSM "waiting for reply" state.  Return the remote exception as a
            # normal reply so the real controller alarm reaches the operator.
            request = None
            try:
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    # log_write(str(self.port) + ": get_joint_state start")
                    result = self._robot.get_joint_state()
                    # log_write(str(self.port) + ": get_joint_state end")
                elif method == "command_joint_state":
                    tic = time.time()
                    toc = time.time()
                    # log_write(str(self.port) + ": command_joint_state start")
                    result = self._robot.command_joint_state(**args)
                    # log_write(str(self.port) + ": command_joint_state end")
                elif method == "get_observations":
                    result = self._robot.get_observations()
                elif method == "set_do_status":
                    result = self._robot.set_do_status(**args)
                elif method == "get_XYZrxryrz_state":
                    result = self._robot.get_XYZrxryrz_state()
                else:
                    raise NotImplementedError(
                        f"Invalid method: {method}, args={args}"
                    )
                response = result
            except Exception as exc:
                response = {
                    _REMOTE_ERROR_KEY: {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "method": request.get("method") if isinstance(request, dict) else None,
                    }
                }
                print(
                    f"[ROBOT SERVER] {type(exc).__name__}: {exc}",
                    flush=True,
                )

            self._socket.send(pickle.dumps(response))

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQClientRobot(Robot):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(
        self,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
        timeout_ms: int = 5000,
    ):
        self._context = zmq.Context()
        self._timeout_ms = timeout_ms
        self._endpoint = f"tcp://{host}:{port}"
        self._lock = threading.Lock()
        self._closed = False
        self._socket = self._open_socket()

    def _open_socket(self):
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self._endpoint)
        return socket

    def _replace_socket(self):
        """Restore the REQ send/receive state after a failed transaction.

        Retrying a robot command automatically is unsafe because the server may
        have executed it before its reply was lost.  We therefore reconnect for
        the *next* request and still report the current request as failed.
        """
        old_socket = self._socket
        self._socket = None
        if old_socket is not None:
            old_socket.close(linger=0)
        if not self._closed:
            self._socket = self._open_socket()

    def _request(self, method: str, args=None):
        request = {"method": method}
        if args is not None:
            request["args"] = args
        with self._lock:
            if self._closed:
                raise RuntimeError("robot client is closed")
            try:
                self._socket.send(pickle.dumps(request))
                result = pickle.loads(self._socket.recv())
            except zmq.Again as exc:
                self._replace_socket()
                raise TimeoutError(
                    f"robot server method {method!r} at {self._endpoint} "
                    "did not reply within "
                    f"{self._timeout_ms / 1000:g}s; inspect launch_nodes.py "
                    "and the Dobot controller alarm"
                ) from exc
            except zmq.ZMQError as exc:
                self._replace_socket()
                if exc.errno == zmq.EFSM:
                    raise ConnectionError(
                        f"robot ZMQ request {method!r} had an invalid state at "
                        f"{self._endpoint}; "
                        "the socket has been reconnected"
                    ) from exc
                raise

            if isinstance(result, dict) and _REMOTE_ERROR_KEY in result:
                error = result[_REMOTE_ERROR_KEY]
                raise RobotServerError(
                    f"robot server {error.get('method')!r} failed with "
                    f"{error.get('type', 'Exception')}: {error.get('message', '')}"
                )
            return result

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        return self._request("num_dofs")

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        return self._request("get_joint_state")

    def command_joint_state(self, joint_state: np.ndarray, flag_in) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
            flag_in
        """
        return self._request(
            "command_joint_state",
            {"joint_state": joint_state, "flag_in": flag_in},
        )

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        return self._request("get_observations")

    def set_do_status(self, which_do: np.ndarray):
        return self._request("set_do_status", {"which_do": which_do})

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._socket is not None:
                self._socket.close(linger=0)
                self._socket = None
            self._context.term()

    def get_XYZrxryrz_state(self):
        return self._request("get_XYZrxryrz_state")
