# _max_cyan_ — project_mxsa
"""behavior trainer — trains emotion/behavior decision model.

creates a decision tree classifier that maps (emotion_state, sensor_inputs)
to action types. training data is generated programmatically from rule-based
scenarios reflecting simba's personality.
"""

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import os
import logging
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# project root
_project_root = Path(__file__).resolve().parent.parent

# load config
_config_path = _project_root / "config" / "simba_config.yaml"
try:
    with open(_config_path, "r") as _f:
        _config = yaml.safe_load(_f)
except FileNotFoundError:
    _config = {"ai": {}, "robot": {}}


# --- feature encoding ---

# emotion states (encoded as integers)
EMOTION_MAP = {
    "happy": 0,
    "sad": 1,
    "curious": 2,
    "excited": 3,
    "love": 4,
    "neutral": 5,
}

# sensor input columns:
# [emotion_idx, object_detected, owner_present, obstacle_near,
#  battery_level, time_idle, voice_command]
#
# object_detected: 0=none, 1=known_object, 2=unknown_object
# owner_present: 0=no, 1=yes
# obstacle_near: 0=no, 1=yes
# battery_level: 0=low, 1=medium, 2=high
# time_idle: 0=just_active, 1=short_idle, 2=long_idle
# voice_command: 0=none, 1=greeting, 2=fetch, 3=come, 4=stop, 5=dance

# action types (classification targets)
ACTION_MAP = {
    "idle": 0,
    "greet": 1,
    "wave": 2,
    "grab": 3,
    "fetch": 4,
    "scan": 5,
    "approach": 6,
    "retreat": 7,
    "dance": 8,
    "wiggle": 9,
    "droop": 10,
    "seek_charger": 11,
    "roam": 12,
    "handshake": 13,
    "stop": 14,
}

# inverse maps for label decoding
EMOTION_NAMES = {v: k for k, v in EMOTION_MAP.items()}
ACTION_NAMES = {v: k for k, v in ACTION_MAP.items()}


def _generate_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """generate rule-based training scenarios.

    features: [emotion_idx, object_detected, owner_present, obstacle_near,
               battery_level, time_idle, voice_command]

    returns:
        tuple of (features, labels) as numpy arrays.
    """
    scenarios = []

    # --- greeting scenarios ---
    # owner present + greeting command -> greet/wave/handshake
    # Note: happy(0) and love(4) are excluded here because they
    # have specific handshake behavior defined below (lines ~198-199)
    for emotion in range(6):
        if emotion in (EMOTION_MAP["happy"], EMOTION_MAP["love"]):
            continue  # these emotions use handshake instead
        # voice greeting, owner present
        scenarios.append(([emotion, 0, 1, 0, 2, 0, 1], "greet"))
        scenarios.append(([emotion, 0, 1, 0, 1, 0, 1], "greet"))
        scenarios.append(([emotion, 0, 1, 0, 2, 1, 1], "wave"))

    # love + owner present -> wiggle
    scenarios.append(([EMOTION_MAP["love"], 0, 1, 0, 2, 0, 0], "wiggle"))
    scenarios.append(([EMOTION_MAP["love"], 0, 1, 0, 1, 0, 0], "wiggle"))
    scenarios.append(([EMOTION_MAP["love"], 0, 1, 0, 2, 1, 0], "wiggle"))
    scenarios.append(([EMOTION_MAP["love"], 1, 1, 0, 2, 0, 0], "wiggle"))

    # excited + owner present -> dance/wiggle
    scenarios.append(([EMOTION_MAP["excited"], 0, 1, 0, 2, 0, 0], "dance"))
    scenarios.append(([EMOTION_MAP["excited"], 0, 1, 0, 1, 0, 0], "wiggle"))
    scenarios.append(([EMOTION_MAP["excited"], 1, 1, 0, 2, 0, 0], "dance"))

    # --- fetch scenarios ---
    # voice fetch command + known object -> fetch
    for emotion in range(6):
        scenarios.append(([emotion, 1, 1, 0, 2, 0, 2], "fetch"))
        scenarios.append(([emotion, 1, 1, 0, 1, 0, 2], "fetch"))
        # fetch but no object visible -> scan first
        scenarios.append(([emotion, 0, 1, 0, 2, 0, 2], "scan"))
        scenarios.append(([emotion, 0, 1, 0, 1, 0, 2], "scan"))

    # --- come command -> approach ---
    for emotion in range(6):
        scenarios.append(([emotion, 0, 1, 0, 2, 0, 3], "approach"))
        scenarios.append(([emotion, 0, 1, 0, 1, 0, 3], "approach"))

    # --- stop command -> stop ---
    for emotion in range(6):
        scenarios.append(([emotion, 0, 1, 0, 2, 0, 4], "stop"))
        scenarios.append(([emotion, 0, 0, 0, 2, 0, 4], "stop"))

    # --- dance command -> dance ---
    for emotion in range(6):
        scenarios.append(([emotion, 0, 1, 0, 2, 0, 5], "dance"))
        scenarios.append(([emotion, 0, 1, 0, 1, 0, 5], "dance"))

    # --- object detection scenarios ---
    # curious + unknown object -> scan/approach
    scenarios.append(([EMOTION_MAP["curious"], 2, 0, 0, 2, 0, 0], "scan"))
    scenarios.append(([EMOTION_MAP["curious"], 2, 0, 0, 2, 1, 0], "approach"))
    scenarios.append(([EMOTION_MAP["curious"], 2, 1, 0, 2, 0, 0], "scan"))
    scenarios.append(([EMOTION_MAP["curious"], 1, 0, 0, 2, 0, 0], "approach"))

    # happy + known object -> approach/grab
    scenarios.append(([EMOTION_MAP["happy"], 1, 0, 0, 2, 0, 0], "approach"))
    scenarios.append(([EMOTION_MAP["happy"], 1, 1, 0, 2, 0, 0], "grab"))
    scenarios.append(([EMOTION_MAP["happy"], 1, 0, 0, 1, 0, 0], "approach"))

    # --- obstacle scenarios ---
    # obstacle near -> retreat regardless of emotion
    for emotion in range(6):
        scenarios.append(([emotion, 0, 0, 1, 2, 0, 0], "retreat"))
        scenarios.append(([emotion, 0, 0, 1, 1, 0, 0], "retreat"))
        scenarios.append(([emotion, 1, 0, 1, 2, 0, 0], "retreat"))

    # --- battery scenarios ---
    # low battery -> seek charger
    for emotion in range(6):
        scenarios.append(([emotion, 0, 0, 0, 0, 0, 0], "seek_charger"))
        scenarios.append(([emotion, 0, 1, 0, 0, 0, 0], "seek_charger"))
        scenarios.append(([emotion, 0, 0, 0, 0, 1, 0], "seek_charger"))
        scenarios.append(([emotion, 1, 0, 0, 0, 0, 0], "seek_charger"))

    # --- idle scenarios ---
    # long idle + curious -> roam
    scenarios.append(([EMOTION_MAP["curious"], 0, 0, 0, 2, 2, 0], "roam"))
    scenarios.append(([EMOTION_MAP["curious"], 0, 0, 0, 1, 2, 0], "roam"))
    scenarios.append(([EMOTION_MAP["neutral"], 0, 0, 0, 2, 2, 0], "roam"))
    scenarios.append(([EMOTION_MAP["neutral"], 0, 0, 0, 1, 2, 0], "roam"))

    # happy/neutral short idle -> idle
    scenarios.append(([EMOTION_MAP["happy"], 0, 0, 0, 2, 1, 0], "idle"))
    scenarios.append(([EMOTION_MAP["neutral"], 0, 0, 0, 2, 1, 0], "idle"))
    scenarios.append(([EMOTION_MAP["happy"], 0, 0, 0, 2, 0, 0], "idle"))
    scenarios.append(([EMOTION_MAP["neutral"], 0, 0, 0, 2, 0, 0], "idle"))

    # sad -> droop
    scenarios.append(([EMOTION_MAP["sad"], 0, 0, 0, 2, 0, 0], "droop"))
    scenarios.append(([EMOTION_MAP["sad"], 0, 0, 0, 1, 0, 0], "droop"))
    scenarios.append(([EMOTION_MAP["sad"], 0, 0, 0, 2, 1, 0], "droop"))
    scenarios.append(([EMOTION_MAP["sad"], 0, 1, 0, 2, 0, 0], "droop"))

    # sad + owner greeting -> greet (overrides droop)
    scenarios.append(([EMOTION_MAP["sad"], 0, 1, 0, 2, 0, 1], "greet"))

    # excited + long idle -> scan
    scenarios.append(([EMOTION_MAP["excited"], 0, 0, 0, 2, 2, 0], "scan"))
    scenarios.append(([EMOTION_MAP["excited"], 0, 0, 0, 1, 2, 0], "scan"))

    # --- handshake scenarios ---
    # greeting + owner present + happy/love -> handshake
    scenarios.append(([EMOTION_MAP["happy"], 0, 1, 0, 2, 0, 1], "handshake"))
    scenarios.append(([EMOTION_MAP["love"], 0, 1, 0, 2, 0, 1], "handshake"))

    # build arrays
    features = np.array([s[0] for s in scenarios], dtype=np.float32)
    label_strings = [s[1] for s in scenarios]
    labels = np.array(
        [ACTION_MAP[a] for a in label_strings], dtype=np.int32
    )

    return features, labels


class BehaviorTrainer:
    """trains a decision tree behavior model for simba.

    maps (emotion_state, sensor_inputs) to action types using a
    decision tree classifier trained on programmatically generated
    rule-based scenarios.
    """

    def __init__(self):
        """initialize the behavior trainer."""
        self._progress = {"current": 0, "total": 0, "phase": "idle"}

    def create_default_behavior_model(
        self, max_depth: int = 10, cv_folds: int = 2
    ) -> Tuple[DecisionTreeClassifier, Dict]:
        """create and train the default behavior model.

        generates training data from rule-based scenarios and trains
        a decision tree classifier.

        args:
            max_depth: maximum tree depth.
            cv_folds: number of cross-validation folds.

        returns:
            tuple of (trained_model, metrics_dict).
        """
        self._progress = {"current": 0, "total": 3, "phase": "generating data"}

        # generate training data
        features, labels = _generate_training_data()

        if len(features) < 2:
            raise ValueError(
                "not enough training scenarios generated (need at least 2).")

        logger.info(f"generated {len(features)} training scenarios")
        logger.info(f"feature dimensions: {features.shape[1]}")
        logger.info(f"unique actions: {len(set(labels))}")

        self._progress = {"current": 1, "total": 3, "phase": "training tree"}

        gc.collect()

        # train decision tree
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(features, labels)

        # cross-validate
        self._progress = {"current": 2, "total": 3, "phase": "validating"}
        n_folds = min(cv_folds, len(features))
        cv_scores = cross_val_score(
            model, features, labels, cv=n_folds, scoring="accuracy", n_jobs=-1
        )

        # tree statistics
        n_leaves = model.get_n_leaves()
        tree_depth = model.get_depth()

        # feature importances
        feature_names = [
            "emotion", "object_detected", "owner_present",
            "obstacle_near", "battery_level", "time_idle", "voice_command",
        ]
        importances = dict(zip(feature_names, model.feature_importances_))

        metrics = {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "n_scenarios": len(features),
            "n_actions": len(set(labels)),
            "tree_depth": tree_depth,
            "n_leaves": n_leaves,
            "feature_importances": {
                k: round(float(v), 4) for k, v in importances.items()
            },
        }

        logger.info(f"\ncross-validation accuracy: {cv_scores.mean():.4f} "
                    f"(±{cv_scores.std():.4f})")
        logger.info(f"tree depth: {tree_depth}, leaves: {n_leaves}")
        logger.info("\nfeature importances:")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
            logger.info(f"  {name}: {imp:.4f}")

        self._progress = {"current": 3, "total": 3, "phase": "complete"}
        return model, metrics

    def save_model(
        self,
        model: DecisionTreeClassifier,
        output_path: Optional[str] = None,
    ) -> str:
        """save the behavior model to disk.

        args:
            model: trained decision tree classifier.
            output_path: path to save the model. defaults to
                        models/behavior_model.joblib.

        returns:
            path where model was saved.
        """
        if output_path is None:
            output_path = str(
                _project_root /
                "models" /
                "behavior_model.joblib")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(model, output_path)
        logger.info(f"behavior model saved to: {output_path}")
        return output_path

    def predict_action(
        self,
        model: DecisionTreeClassifier,
        emotion: str,
        object_detected: int = 0,
        owner_present: bool = False,
        obstacle_near: bool = False,
        battery_level: int = 2,
        time_idle: int = 0,
        voice_command: int = 0,
    ) -> Tuple[str, float]:
        """predict the best action given current state.

        args:
            model: trained behavior model.
            emotion: current emotion string (e.g., 'happy', 'curious').
            object_detected: 0=none, 1=known, 2=unknown.
            owner_present: whether the owner is present.
            obstacle_near: whether an obstacle is nearby.
            battery_level: 0=low, 1=medium, 2=high.
            time_idle: 0=just_active, 1=short_idle, 2=long_idle.
            voice_command: 0=none, 1=greeting, 2=fetch, 3=come, 4=stop, 5=dance.

        returns:
            tuple of (action_name, confidence).
        """
        emotion_idx = EMOTION_MAP.get(emotion.lower(), EMOTION_MAP["neutral"])
        features = np.array(
            [[emotion_idx, object_detected, int(owner_present),
              int(obstacle_near), battery_level, time_idle, voice_command]],
            dtype=np.float32,
        )

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities.max())

        action_name = ACTION_NAMES.get(int(prediction), "idle")
        return action_name, confidence

    @property
    def progress(self) -> dict:
        """get current training progress for the web ui."""
        return self._progress.copy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="train simba behavior model")
    parser.add_argument(
        "--output",
        default=None,
        help="path to save the behavior model",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="run test predictions after training",
    )
    args = parser.parse_args()

    trainer = BehaviorTrainer()
    model, metrics = trainer.create_default_behavior_model()
    trainer.save_model(model, args.output)

    if args.test:
        print("\n--- test predictions ---")
        test_cases = [
            {"emotion": "happy", "owner_present": True, "voice_command": 1},
            {"emotion": "curious", "object_detected": 2},
            {"emotion": "sad"},
            {"emotion": "excited", "voice_command": 5},
            {"emotion": "neutral", "battery_level": 0},
            {"emotion": "love", "owner_present": True},
            {"emotion": "curious", "time_idle": 2},
        ]
        for case in test_cases:
            action, conf = trainer.predict_action(model, **case)
            print(f"  {case} -> {action} (confidence: {conf:.3f})")
