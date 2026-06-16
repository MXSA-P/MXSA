# _max_cyan_ — project_mxsa
"""robot operational state machine for simba.

manages simba's high-level operational states with validated transitions,
context tracking, state duration monitoring, and transition history.
thread-safe for concurrent access from all subsystems.
"""

import copy
import json
import threading
import time
from typing import Any, Dict, List, Optional, FrozenSet

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.core.state_machine")

# all valid states
_valid_states = frozenset({
    "BOOTING", "SCANNING", "ROAMING", "FETCHING", "DELIVERING",
    "SEARCHING", "PLAYING", "CHARGING", "IDLE", "GREETING", "EXPRESSING",
    "HOLDING", "FOLLOWING", "GRABBING", "ACTING", "RETURNING",
})

# states that indicate the robot is busy with a task
_busy_states = frozenset({"FETCHING", "DELIVERING", "SEARCHING", "HOLDING", "FOLLOWING", "GRABBING", "RETURNING"})

# states that block accepting new commands
_critical_states = frozenset({"BOOTING", "CHARGING"})

# valid state transitions — from_state: set of valid to_states
# most states can transition to IDLE and common interrupt states
_transitions: Dict[str, FrozenSet[str]] = {
    "BOOTING": frozenset({
        "IDLE", "SCANNING", "CHARGING",
    }),
    "IDLE": frozenset({
        "SCANNING", "ROAMING", "FETCHING", "SEARCHING", "PLAYING",
        "CHARGING", "GREETING", "EXPRESSING", "HOLDING", "FOLLOWING",
        "GRABBING", "ACTING", "RETURNING",
    }),
    "SCANNING": frozenset({
        "IDLE", "ROAMING", "FETCHING", "SEARCHING", "CHARGING",
        "GREETING", "EXPRESSING",
    }),
    "ROAMING": frozenset({
        "IDLE", "SCANNING", "FETCHING", "SEARCHING", "PLAYING",
        "CHARGING", "GREETING", "EXPRESSING", "RETURNING",
    }),
    "FETCHING": frozenset({
        "IDLE", "HOLDING", "DELIVERING", "SEARCHING", "SCANNING",
    }),
    "DELIVERING": frozenset({
        "IDLE", "SCANNING", "EXPRESSING", "HOLDING",
    }),
    "SEARCHING": frozenset({
        "IDLE", "FETCHING", "SCANNING", "ROAMING",
    }),
    "PLAYING": frozenset({
        "IDLE", "SCANNING", "GREETING", "EXPRESSING",
    }),
    "CHARGING": frozenset({
        "IDLE", "SCANNING", "EXPRESSING",
    }),
    "GREETING": frozenset({
        "IDLE", "SCANNING", "ROAMING", "EXPRESSING",
    }),
    "EXPRESSING": frozenset({
        "IDLE", "SCANNING", "ROAMING", "FETCHING", "PLAYING",
        "GREETING",
    }),
    "HOLDING": frozenset({
        "IDLE", "SCANNING", "EXPRESSING", "DELIVERING", "ROAMING",
    }),
    "FOLLOWING": frozenset({
        "IDLE", "SCANNING", "EXPRESSING", "GRABBING",
    }),
    "GRABBING": frozenset({
        "IDLE", "HOLDING", "EXPRESSING",
    }),
    "ACTING": frozenset({
        "IDLE", "SCANNING", "EXPRESSING", "ROAMING", "RETURNING"
    }),
    "RETURNING": frozenset({
        "CHARGING", "IDLE",
    }),
}


class StateMachine:
    """robot operational state machine.

    manages high-level states for the simba robot with validated
    transitions, context data, duration tracking, and transition
    history. provides query methods for the brain and web dashboard.

    attributes:
        state: current operational state string.
    """

    def __init__(self) -> None:
        """initialize the state machine in booting state."""
        self._lock = threading.Lock()
        self._state: str = "BOOTING"
        self._context: Dict[str, Any] = {}
        self._state_start: float = time.time()
        self._history: List[Dict[str, Any]] = []
        self._max_history: int = 100
        self._transition_count: int = 0

        logger.info("state machine initialized in BOOTING state")
        log_event("state", "state machine initialized", {"state": "BOOTING"})

    def transition(
        self, new_state: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """transition to a new state with optional context data.

        validates that the transition is allowed from the current state.
        records the transition in history and logs it.

        args:
            new_state: the target state to transition to.
            context: optional dict of context data for the new state
                    (e.g., target object name for fetching).

        returns:
            true if the transition succeeded, false if invalid.
        """
        new_state = new_state.upper()

        if new_state not in _valid_states:
            logger.error("invalid state '%s'", new_state)
            return False

        with self._lock:
            old_state = self._state

            if old_state == new_state:
                # allow context updates within the same state
                if context is not None:
                    self._context.update(context)
                return True

            # validate transition
            allowed = _transitions.get(old_state, frozenset())
            if new_state not in allowed:
                logger.warning("invalid transition %s -> %s (allowed: %s)",
                               old_state, new_state,
                               ", ".join(sorted(allowed)))
                return False

            # calculate duration
            now = time.time()
            duration = now - self._state_start

            # record transition history
            self._history.append({
                "from_state": old_state,
                "to_state": new_state,
                "timestamp": now,
                "duration_in_previous": round(duration, 2),
                "context": copy.deepcopy(self._context),
            })
            if len(self._history) > self._max_history:
                self._history.pop(0)

            # perform transition
            self._state = new_state
            self._context = copy.deepcopy(context) if context else {}
            self._state_start = now
            self._transition_count += 1

        logger.info("state transition: %s -> %s (was in %s for %.1fs)",
                    old_state, new_state, old_state, duration)
        log_event("state", f"{old_state} -> {new_state}", {
            "from": old_state,
            "to": new_state,
            "duration": round(duration, 2),
            "context": context or {},
        })

        return True

    def get_state(self) -> str:
        """get the current operational state.

        returns:
            current state string (uppercase).
        """
        with self._lock:
            return self._state

    def get_state_duration(self) -> float:
        """get how long the robot has been in the current state.

        returns:
            seconds spent in the current state, or 0.0 if
            the entered_at timestamp is unavailable.
        """
        with self._lock:
            if hasattr(self, "_state_start"):
                return time.time() - self._state_start
            return 0.0

    def get_context(self) -> Dict[str, Any]:
        """get the context data for the current state.

        returns:
            copy of the current state context dict.
        """
        with self._lock:
            return copy.deepcopy(self._context)

    def get_history(self, count: int = 10) -> List[Dict[str, Any]]:
        """get recent state transition history.

        args:
            count: maximum number of transitions to return.

        returns:
            list of transition dicts (most recent last).
        """
        with self._lock:
            return copy.deepcopy(self._history[-count:])

    def is_busy(self) -> bool:
        """check if the robot is busy with a task.

        returns:
            true if in fetching, delivering, or searching state.
        """
        with self._lock:
            return self._state in _busy_states

    def can_accept_command(self) -> bool:
        """check if the robot can accept a new command.

        returns:
            true unless in a critical state (booting, charging).
        """
        with self._lock:
            return self._state not in _critical_states

    def to_dict(self) -> Dict[str, Any]:
        """serialize state machine data for the web dashboard.

        returns:
            dict with state, duration, context, transition_count,
            is_busy, can_accept_command, and recent history.
        """
        with self._lock:
            state = self._state
            duration = time.time() - self._state_start
            context = copy.deepcopy(self._context)
            count = self._transition_count
            history = copy.deepcopy(self._history[-10:])

        return {
            "state": state,
            "duration": round(duration, 2),
            "context": context,
            "transition_count": count,
            "is_busy": state in _busy_states,
            "can_accept_command": state not in _critical_states,
            "history": history,
        }

    def dump_history(self, filepath: str) -> None:
        """dump full state transition history to a file.
        
        args:
            filepath: path to the output json file.
        """
        with self._lock:
            history_copy = list(self._history)
        try:
            with open(filepath, "w") as f:
                json.dump(history_copy, f, indent=2)
            logger.info(f"dumped state history to {filepath}")
        except Exception as e:
            logger.error(f"failed to dump state history: {e}")

    def dump_history_json(self) -> str:
        """return full state transition history as a json string.

        returns:
            json string of the transition history list,
            or '[]' if dump_history is unavailable.
        """
        if hasattr(self, "dump_history"):
            with self._lock:
                return json.dumps(list(self._history))
        return "[]"

    def __repr__(self) -> str:
        with self._lock:
            state = self._state
            duration = time.time() - self._state_start
            transitions = self._transition_count
        return (
            f"StateMachine(state='{state}', "
            f"duration={duration:.1f}s, "
            f"transitions={transitions})"
        )
