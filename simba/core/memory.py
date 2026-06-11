# _max_cyan_ — project_mxsa
"""persistent object memory database for simba robot.

stores information about objects that simba has seen, including
their position angles, confidence scores, embeddings, and timestamps.
data is persisted to a json file and auto-saved after every change.
"""

import json
import os
import time
import tempfile
import threading
import copy
from datetime import datetime
from typing import Any, Dict, List, Optional

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.core.memory")

class NumpyEncoder(json.JSONEncoder):
    """custom encoder to handle numpy arrays in json serialization."""
    def default(self, obj: Any) -> Any:
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


class MemorySystem:
    """persistent object memory database.

    maintains a dictionary of remembered objects with metadata
    including position, confidence, timestamps, and optional
    feature embeddings. data is auto-saved to a json file
    after every mutation.

    attributes:
        db_path: absolute path to the json database file.
        max_items: maximum number of objects to store.
    """

    def __init__(self, config: dict) -> None:
        """initialize the memory system.

        args:
            config: full simba configuration dict loaded from yaml.
        """
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._objects: Dict[str, Dict[str, Any]] = {}
        self._dirty: bool = False

        ai_cfg = config.get("ai", {})
        self.max_items: int = ai_cfg.get("max_memory_items", 500)

        # resolve database path relative to project root
        db_path = ai_cfg.get("memory_db_path", "data/memory.json")
        if not os.path.isabs(db_path):
            project_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ))
            db_path = os.path.join(project_root, db_path)
        self.db_path: str = db_path

        # load existing memory
        self.load()

        log_event("memory", "memory system initialized", {
            "objects_loaded": len(self._objects),
            "db_path": self.db_path,
            "max_items": self.max_items,
        })

    def remember_object(
        self,
        label: str,
        position_angle: float,
        confidence: float,
        embedding: Optional[List[float]] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> None:
        """store or update an object in memory with 2D spatial mapping.

        args:
            label: object label (normalized to lowercase).
            position_angle: angle in degrees where the object was seen.
            confidence: detection confidence score (0.0 to 1.0).
            embedding: optional feature embedding vector.
            x: horizontal coordinate on the floor (cm)
            y: vertical (depth) coordinate on the floor (cm)
        """
        label = label.strip().lower()
        now = datetime.now().isoformat()

        with self._lock:
            if label in self._objects:
                entry = self._objects[label]
                entry["position_angle"] = position_angle
                entry["confidence"] = confidence
                entry["last_seen_timestamp"] = now
                entry["times_seen"] = entry.get("times_seen", 0) + 1
                if embedding is not None:
                    entry["embedding"] = list(embedding)
                if x is not None:
                    entry["x"] = x
                if y is not None:
                    entry["y"] = y
                logger.debug("updated object '%s' at (%.1f, %.1f) "
                             "(seen %d times)", label, x or 0, y or 0,
                             entry["times_seen"])
            else:
                while len(self._objects) >= max(1, self.max_items):
                    self._evict_oldest()

                self._objects[label] = {
                    "label": label,
                    "position_angle": position_angle,
                    "confidence": confidence,
                    "last_seen_timestamp": now,
                    "first_seen_timestamp": now,
                    "times_seen": 1,
                    "embedding": list(embedding) if embedding is not None else None,
                    "x": x,
                    "y": y,
                }
                logger.info("remembered new object '%s' at angle %.1f",
                            label, position_angle)

            self._dirty = True

        self.save()
        log_event("memory", f"remembered object: {label}", {
            "label": label,
            "position_angle": position_angle,
            "confidence": round(confidence, 3),
        })

    def forget_object(self, label: str) -> bool:
        """remove an object from memory.

        args:
            label: object label to forget.

        returns:
            true if the object was found and removed, false otherwise.
        """
        label = label.strip().lower()

        with self._lock:
            if label in self._objects:
                del self._objects[label]
                self._dirty = True
                logger.info("forgot object '%s'", label)
            else:
                logger.debug("object '%s' not in memory", label)
                return False

        self.save()
        log_event("memory", f"forgot object: {label}", {"label": label})
        return True

    def find_object(self, label: str) -> Optional[Dict[str, Any]]:
        """find an object in memory by label.

        args:
            label: object label to search for.

        returns:
            dict with object data (label, position_angle, confidence,
            last_seen_timestamp, times_seen, embedding) or none.
        """
        label = label.strip().lower()

        with self._lock:
            entry = self._objects.get(label)
            if entry is not None:
                # return a deep copy to prevent external mutation
                return copy.deepcopy(entry)
        return None

    def get_all_objects(self) -> List[Dict[str, Any]]:
        """get all remembered objects.

        returns:
            list of object dicts sorted by last_seen_timestamp descending.
        """
        with self._lock:
            objects = [copy.deepcopy(obj) for obj in self._objects.values()]

        objects.sort(
            key=lambda o: o.get("last_seen_timestamp", ""),
            reverse=True,
        )
        return objects

    def get_charging_pad(self) -> Optional[Dict[str, Any]]:
        """find the charging pad location in memory.

        looks for objects with labels matching 'charging_pad',
        'charger', or 'charging pad'.

        returns:
            dict with charging pad data, or none if not found.
        """
        pad_labels = ["charging_pad", "charger", "charging pad"]
        with self._lock:
            for pad_label in pad_labels:
                entry = self._objects.get(pad_label)
                if entry is not None:
                    return copy.deepcopy(entry)
        return None

    def is_known(self, label: str) -> bool:
        """check if an object is in memory.

        args:
            label: object label to check.

        returns:
            true if the object is remembered.
        """
        label = label.strip().lower()
        with self._lock:
            return label in self._objects

    def save(self) -> bool:
        """persist memory to the json file.

        returns:
            true if save succeeded, false otherwise.
        """
        with self._save_lock:
            with self._lock:
                if not self._dirty:
                    return True
                # deep copy to safely format without mutating actual state
                data = copy.deepcopy(self._objects)
                self._dirty = False
    
            # disk I/O outside main lock but inside save lock
            tmp_path = None
            try:
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(self.db_path), prefix="memory_", suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False, cls=NumpyEncoder)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, self.db_path)
                try:
                    dir_fd = os.open(os.path.dirname(self.db_path), os.O_RDONLY)
                    os.fsync(dir_fd)
                    os.close(dir_fd)
                except OSError:
                    pass
    
                logger.debug("memory saved (%d objects)", len(data))
                return True
    
            except Exception as exc:
                logger.error("failed to save memory: %s", exc)
                with self._lock:
                    self._dirty = True
                return False
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    def load(self) -> bool:
        """load memory from the json file.

        returns:
            true if load succeeded (or file doesn't exist yet), false on error.
        """
        if not os.path.isfile(self.db_path):
            logger.info("no memory file found at %s — starting fresh",
                        self.db_path)
            return True

        try:
            with open(self.db_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            with self._lock:
                self._objects = {}
                for label, entry in data.items():
                    self._objects[label.strip().lower()] = entry
                while len(self._objects) > self.max_items:
                    self._evict_oldest()
                self._dirty = False

            logger.info("memory loaded: %d objects from %s",
                        len(self._objects), self.db_path)
            return True

        except Exception as exc:
            logger.error("failed to load memory (%s): %s", type(exc).__name__, exc)
            backup_path = self.db_path + ".bak"
            try:
                if os.path.exists(self.db_path):
                    os.rename(self.db_path, backup_path)
                    logger.info("Backed up corrupted memory file to %s", backup_path)
            except Exception as e:
                logger.error("Failed to back up corrupted memory file: %s", e)
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """get statistics about the memory database.

        returns:
            dict with count, oldest timestamp, and newest timestamp.
        """
        with self._lock:
            count = len(self._objects)

            if count == 0:
                return {
                    "count": 0,
                    "oldest": None,
                    "newest": None,
                }

            timestamps = [
                obj.get("last_seen_timestamp", "")
                for obj in self._objects.values()
                if obj.get("last_seen_timestamp")
            ]

            return {
                "count": count,
                "oldest": min(timestamps) if timestamps else None,
                "newest": max(timestamps) if timestamps else None,
            }

    def _evict_oldest(self) -> None:
        """remove the oldest (least recently seen) object to make room.

        called internally when capacity is reached. must be called
        while holding self._lock.
        """
        if not self._objects:
            return

        oldest_label = min(
            self._objects,
            key=lambda k: self._objects[k].get("last_seen_timestamp", ""),
        )
        logger.info("evicting oldest object '%s' to make room", oldest_label)
        del self._objects[oldest_label]

    def decay(self, max_age_seconds: float = 86400) -> None:
        """remove objects older than max_age_seconds."""
        now = datetime.now()
        removed = 0
        with self._lock:
            for label in list(self._objects.keys()):
                entry = self._objects[label]
                try:
                    last_seen = datetime.fromisoformat(
                        entry["last_seen_timestamp"])
                    if (now - last_seen).total_seconds() > max_age_seconds:
                        del self._objects[label]
                        removed += 1
                        self._dirty = True
                except (ValueError, TypeError, KeyError):
                    pass
        if removed > 0:
            logger.info("decayed %d old memory objects", removed)
            self.save()

    def flush(self) -> None:
        """force persistent database flush."""
        logger.info("forcing memory flush")
        self.save()

    def __repr__(self) -> str:
        return (
            f"MemorySystem(objects={len(self._objects)}, "
            f"db_path='{self.db_path}')"
        )
