# _max_cyan_ — project_mxsa
"""path recorder — dead-reckoning breadcrumb trail for return-to-base.

records every chassis movement (action, speed, duration) in a lifo stack.
when "go charge" is triggered, the stack is popped in reverse with each
action inverted (forward↔backward, left↔right) to retrace the exact
path back to the charging station.
"""

import json
import os
import time
import threading
from typing import List, Dict, Any  # Cleaned typing

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.core.path_recorder")

# inverse action mapping — what to do to "undo" each movement
_INVERSE_MAP = {
    "forward": "backward",
    "backward": "forward",
    "left": "right",
    "right": "left",
    "spin_left": "spin_right",
    "spin_right": "spin_left",
    "turn_left": "turn_right",
    "turn_right": "turn_left",
}


class PathRecorder:
    """records chassis movements and replays them in reverse to return home.

    usage:
        recorder = PathRecorder()
        recorder.record("forward", speed=60, duration=2.0)
        recorder.record("spin_left", speed=45, duration=1.0)
        recorder.record("forward", speed=60, duration=2.0)

        # later, to return:
        recorder.replay(chassis)  # executes: backward, spin_right, backward
    """

    def __init__(self, save_path: str = "data/path_log.json",
                 max_entries: int = 2000) -> None:
        self._stack: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._save_path = save_path
        self._max_entries = max_entries
        self._returning = False

        # load any persisted path from a previous session
        self._load()
        logger.info(f"path recorder initialized ({len(self._stack)} entries loaded)")

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------

    def record(self, action: str, speed: float, duration: float,
               imu_tilt: float = 0.0) -> None:
        """push a movement onto the path stack.

        args:
            action: movement type (forward, backward, left, right, spin_left, spin_right)
            speed: motor speed (0-100)
            duration: how long the movement lasted (seconds)
            imu_tilt: optional imu tilt angle at time of movement
        """
        if self._returning:
            # don't record movements made during the return trip
            return

        if action not in _INVERSE_MAP:
            logger.warning(f"unknown action '{action}' — not recording")
            return

        entry = {
            "action": action,
            "speed": speed,
            "duration": round(duration, 3),
            "imu_tilt": round(imu_tilt, 2),
            "timestamp": time.time(),
        }

        with self._lock:
            self._stack.append(entry)

            # cap the stack to prevent memory bloat
            if len(self._stack) > self._max_entries:
                # drop oldest entries (front of list)
                overflow = len(self._stack) - self._max_entries
                self._stack = self._stack[overflow:]
                logger.warning(f"path stack capped at {self._max_entries} entries")

        self._save()
        log_event("navigation", f"recorded: {action} spd={speed} dur={duration:.1f}s")

    # ------------------------------------------------------------------
    # replay (return to charging station)
    # ------------------------------------------------------------------

    def replay(self, chassis, on_step=None) -> bool:
        """pop all movements and execute them in reverse to return home.

        args:
            chassis: ChassisController instance to drive the motors.
            on_step: optional callback(step_num, total, action) for progress.

        returns:
            true if replay completed, false if stack was empty or interrupted.
        """
        with self._lock:
            if not self._stack:
                logger.info("path stack is empty — already at home")
                return True
            # take a snapshot and clear
            path = list(self._stack)

        self._returning = True
        total = len(path)
        logger.info(f"starting return journey: {total} steps to retrace")
        log_event("navigation", f"returning home — {total} steps")

        try:
            for i, entry in enumerate(reversed(path)):
                if not self._returning:
                    logger.info("return journey cancelled")
                    return False

                inverse_action = _INVERSE_MAP.get(entry["action"])
                if not inverse_action:
                    continue

                speed = entry["speed"]
                duration = entry["duration"]

                if on_step:
                    on_step(i + 1, total, inverse_action)

                logger.info(
                    f"step {i + 1}/{total}: {inverse_action} "
                    f"spd={speed} dur={duration:.1f}s"
                )

                # execute the inverse movement
                self._execute_movement(chassis, inverse_action, speed, duration)

            # successfully returned — clear the stack
            with self._lock:
                self._stack.clear()
            self._save()

            logger.info("return journey complete — path stack cleared")
            log_event("navigation", "returned to charging station")
            return True

        except Exception as e:
            logger.error(f"return journey failed: {e}")
            chassis.stop()
            return False
        finally:
            self._returning = False

    def cancel_return(self) -> None:
        """cancel an in-progress return journey."""
        self._returning = False

    # ------------------------------------------------------------------
    # movement execution
    # ------------------------------------------------------------------

    @staticmethod
    def _execute_movement(chassis, action: str, speed: float,
                          duration: float) -> None:
        """execute a single chassis movement for a given duration."""
        action_map = {
            "forward": chassis.forward,
            "backward": chassis.backward,
            "left": chassis.turn_left,
            "right": chassis.turn_right,
            "spin_left": chassis.spin_left,
            "spin_right": chassis.spin_right,
            "turn_left": chassis.turn_left,
            "turn_right": chassis.turn_right,
        }

        move_fn = action_map.get(action)
        if move_fn:
            move_fn(speed=speed)
            time.sleep(duration)
            chassis.stop()
            time.sleep(0.2)  # brief settling pause between moves

    # ------------------------------------------------------------------
    # state queries
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """number of recorded movements in the stack."""
        with self._lock:
            return len(self._stack)

    @property
    def is_returning(self) -> bool:
        """true if currently executing a return journey."""
        return self._returning

    def to_dict(self) -> Dict[str, Any]:
        """serialize state for web dashboard."""
        with self._lock:
            return {
                "step_count": len(self._stack),
                "is_returning": self._returning,
                "last_action": self._stack[-1]["action"] if self._stack else None,
            }

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """persist the path stack to disk."""
        try:
            os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
            with open(self._save_path, "w") as f:
                json.dump(self._stack, f)
        except Exception as e:
            logger.warning(f"failed to save path log: {e}")

    def _load(self) -> None:
        """load the path stack from disk."""
        try:
            if os.path.exists(self._save_path):
                with open(self._save_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._stack = data
                    logger.info(f"loaded {len(self._stack)} path entries from disk")
        except Exception as e:
            logger.warning(f"failed to load path log: {e}")
            self._stack = []

    def clear(self) -> None:
        """manually clear the path stack."""
        with self._lock:
            self._stack.clear()
        self._save()
        logger.info("path stack manually cleared")
