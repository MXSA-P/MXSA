# _max_cyan_ — project_mxsa
"""structured logging for simba robot."""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import threading
import copy
from collections import deque

_max_history = 200
_log_history = deque(maxlen=_max_history)
_log_lock = threading.Lock()
_config_lock = threading.Lock()
_handlers_configured = False


class SimbaFormatter(logging.Formatter):
    """custom formatter with beautiful ANSI colored level names and simba branding."""

    colors = {
        "DEBUG": "\033[38;5;14m",     # bright cyan
        "INFO": "\033[38;5;46m",      # bright green
        "WARNING": "\033[38;5;226m",  # bright yellow
        "ERROR": "\033[38;5;196m",    # bright red
        "CRITICAL": "\033[38;5;199m\033[1m",  # bright pink, bold
    }
    reset = "\033[0m"
    dim = "\033[2m"

    def format(self, record: logging.LogRecord) -> str:
        """format the log record with colored level names and module names.
        
        args:
            record: the log record to format.
            
        returns:
            formatted string.
        """
        orig_levelname = record.levelname
        orig_name = record.name
        orig_msg = record.msg

        color = self.colors.get(record.levelname, self.reset)
        record.levelname = f"{color}{record.levelname:<8}{self.reset}"
        record.name = f"\033[38;5;33m{record.name}\033[0m"

        if orig_levelname in ["ERROR", "CRITICAL"]:
            record.msg = f"{color}{record.msg}{self.reset}"

        result = super().format(record)

        # Apply dim to timestamp
        time_str = self.formatTime(record, self.datefmt)
        result = result.replace(f"[{time_str}]",
                                f"{self.dim}[{time_str}]{self.reset}")

        # Restore original values
        record.levelname = orig_levelname
        record.name = orig_name
        record.msg = orig_msg

        return result


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    global _handlers_configured
    logger = logging.getLogger(name)

    if not _handlers_configured:
        with _config_lock:
            if not _handlers_configured:
                root_simba = logging.getLogger("simba")
                root_simba.setLevel(level)

                # console handler
                console = logging.StreamHandler(sys.stdout)
                console.setLevel(level)
                fmt = SimbaFormatter(
                    "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
                    datefmt="%H:%M:%S"
                )
                console.setFormatter(fmt)
                root_simba.addHandler(console)

                # file handler
                log_dir = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )), "data", "logs")
                os.makedirs(log_dir, exist_ok=True)

                log_file = os.path.join(
                    log_dir,
                    "simba_system.jsonl"
                )
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=5 * 1024 * 1024, backupCount=3
                )
                file_handler.setLevel(logging.DEBUG)

                class JSONFormatter(logging.Formatter):
                    """Structured JSON formatter for file logging."""

                    def format(self, record):
                        log_record = {
                            "timestamp": self.formatTime(record, self.datefmt),
                            "level": record.levelname,
                            "name": record.name,
                            "message": record.getMessage(),
                        }
                        if record.exc_info:
                            log_record["exception"] = self.formatException(
                                record.exc_info)
                        return json.dumps(log_record)

                file_fmt = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
                file_handler.setFormatter(file_fmt)
                root_simba.addHandler(file_handler)
                
                _handlers_configured = True

    return logger


def log_event(category: str, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """log a structured event for the web dashboard.

    args:
        category: event category (e.g., 'vision', 'voice', 'motion')
        message: human-readable message
        data: optional dict of additional data
        
    returns:
        the created event dict
    """
    event = {
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "message": message,
        "data": copy.deepcopy(data) if data else {}
    }
    with _log_lock:
        _log_history.append(event)

    return event


def get_log_history(count: int = 50) -> List[Dict[str, Any]]:
    """get recent log events for the web dashboard.

    args:
        count: number of recent events to return

    returns:
        list of event dicts
    """
    with _log_lock:
        return list(_log_history)[-count:]


def clear_log_history() -> None:
    """clear the in-memory log history."""
    with _log_lock:
        _log_history.clear()
