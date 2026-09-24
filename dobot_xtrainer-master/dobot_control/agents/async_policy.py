"""Single-flight asynchronous policy inference helpers.

The worker is deliberately independent from robot and camera code so its
queueing, timeout, generation, and action-stitching behavior can be tested
without hardware.
"""

import dataclasses
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class InferenceResult:
    request_id: int
    generation: int
    request_step: int
    latency_ms: float
    output: Optional[Dict[str, Any]] = None
    error: Optional[BaseException] = None


class AsyncPolicyRunner:
    """Runs at most one ``policy.infer`` call at a time on a daemon thread."""

    def __init__(self, policy):
        self._policy = policy
        self._requests = queue.Queue(maxsize=1)
        self._results = queue.Queue()
        self._lock = threading.Lock()
        self._busy = False
        self._next_request_id = 1
        self._closed = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="openpi-policy-worker",
            daemon=True,
        )
        self._worker.start()

    def submit(self, observation, generation: int, request_step: int) -> Optional[int]:
        """Submit one request, returning ``None`` if another is in flight."""
        with self._lock:
            if self._closed or self._busy:
                return None
            request_id = self._next_request_id
            self._next_request_id += 1
            self._busy = True
        self._requests.put_nowait(
            (request_id, generation, request_step, time.monotonic(), observation)
        )
        return request_id

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def poll(self) -> List[InferenceResult]:
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def wait(self, request_id: int, timeout_s: float) -> InferenceResult:
        """Wait for a particular request, discarding older completed results."""
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"policy request {request_id} did not finish within {timeout_s:.1f}s"
                )
            try:
                result = self._results.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if result.request_id == request_id:
                return result

    def wait_until_idle(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while self.is_busy() and time.monotonic() < deadline:
            time.sleep(0.01)
        return not self.is_busy()

    def discard_completed(self) -> None:
        self.poll()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            # An in-flight request will finish shortly. The daemon worker must
            # never keep process shutdown waiting on a failed network call.
            pass

    def _worker_loop(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            request_id, generation, request_step, submitted_at, observation = request
            output = None
            error = None
            try:
                output = self._policy.infer(observation)
            except BaseException as exc:  # propagate to the control thread
                error = exc
            latency_ms = (time.monotonic() - submitted_at) * 1000.0
            self._results.put(
                InferenceResult(
                    request_id=request_id,
                    generation=generation,
                    request_step=request_step,
                    latency_ms=latency_ms,
                    output=output,
                    error=error,
                )
            )
            with self._lock:
                self._busy = False


class TemporalActionEnsembler:
    """Average overlapping arm plans that target the same rollout timestep.

    pi0.5 produces a stochastic action chunk on every request.  Replacing the
    active chunk wholesale can turn small disagreements between consecutive
    predictions into a periodic reversal at the replan frequency.  This class
    aligns chunks by their request step and averages a bounded number of plans.
    Newer plans receive exponentially larger weight because they use fresher
    images and robot state.  Gripper commands stay discrete and come from the
    newest plan.
    """

    _ARM_INDICES = np.r_[0:6, 7:13]

    def __init__(self, max_chunks: int = 3, decay: float = 0.7):
        if max_chunks < 1:
            raise ValueError("max_chunks must be at least 1")
        if decay < 0:
            raise ValueError("decay cannot be negative")
        self.max_chunks = int(max_chunks)
        self.decay = float(decay)
        self._plans: List[Tuple[int, np.ndarray]] = []

    def clear(self) -> None:
        self._plans.clear()

    def add(self, actions: np.ndarray, request_step: int) -> None:
        chunk = np.asarray(actions, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 14:
            raise ValueError(f"unexpected action shape {chunk.shape}, want (H, 14)")
        if len(chunk) == 0:
            raise ValueError("policy returned an empty action chunk")
        if request_step < 0:
            raise ValueError("request_step cannot be negative")

        # A request id is single-flight, so duplicate origins should not occur;
        # replacing one makes the helper robust to a replayed network result.
        self._plans = [plan for plan in self._plans if plan[0] != request_step]
        self._plans.append((int(request_step), chunk.copy()))
        self._plans.sort(key=lambda plan: plan[0], reverse=True)
        del self._plans[self.max_chunks :]

    def action_for_step(self, step: int) -> Tuple[np.ndarray, int, int]:
        """Return action, newest-chunk index, and contributor count for ``step``."""
        candidates = []
        for origin, chunk in self._plans:
            index = int(step) - origin
            if 0 <= index < len(chunk):
                candidates.append((origin, index, chunk[index]))
        if not candidates:
            raise IndexError(f"no policy action available for rollout step {step}")

        # Plans are already newest-first.  A decay of 0.7 gives normalized
        # weights 0.57/0.29/0.14 when three predictions overlap.
        weights = np.exp(-self.decay * np.arange(len(candidates), dtype=np.float64))
        newest_action = candidates[0][2]
        action = newest_action.copy()
        arm_values = np.stack([item[2][self._ARM_INDICES] for item in candidates])
        action[self._ARM_INDICES] = np.average(arm_values, axis=0, weights=weights)
        return action, candidates[0][1], len(candidates)


def prepare_action_chunk(
    actions: np.ndarray,
    *,
    steps_elapsed: int,
    last_action: np.ndarray,
    blend_steps: int,
) -> Tuple[np.ndarray, int]:
    """Skip latency-stale actions and blend arm joints at the chunk boundary."""
    chunk = np.asarray(actions, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 14:
        raise ValueError(f"unexpected action shape {chunk.shape}, want (H, 14)")
    if len(chunk) == 0:
        raise ValueError("policy returned an empty action chunk")

    chunk = chunk.copy()
    start_index = min(max(int(steps_elapsed), 0), len(chunk) - 1)
    available = len(chunk) - start_index
    count = min(max(int(blend_steps), 0), available)
    if count:
        arm_indices = np.r_[0:6, 7:13]
        previous = np.asarray(last_action, dtype=np.float64)
        for offset in range(count):
            alpha = (offset + 1) / count
            index = start_index + offset
            chunk[index, arm_indices] = (
                (1.0 - alpha) * previous[arm_indices]
                + alpha * chunk[index, arm_indices]
            )
    return chunk, start_index
