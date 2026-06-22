# _max_cyan_ — project_mxsa
"""vision trainer — object recognition training using mobilenetv2 + svm.

extracts 1280-dimensional feature vectors from mobilenetv2 (tflite) and
trains a calibrated linear svc for object classification. designed to run
on a linux pc for training, then export the svm model to the raspberry pi.
"""

import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import os
import json
import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from ai_edge_litert import interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            tflite = None


# project root is three levels up from this file
_project_root = Path(__file__).resolve().parent.parent

# load config
_config_path = _project_root / "config" / "simba_config.yaml"
try:
    with open(_config_path, "r") as _f:
        _config = yaml.safe_load(_f)
except FileNotFoundError:
    _config = {"ai": {"vision_model_path": "models/mobilenetv2_feature_extractor.tflite"}}


class VisionTrainer:
    """trains an object classifier using mobilenetv2 features + linear svc.

    workflow:
        1. load mobilenetv2 tflite model (feature extractor, outputs 1280-dim)
        2. for each class directory, extract features from all images
        3. train a linearsvc with probability calibration
        4. evaluate with cross-validation
        5. save model + label map for deployment on rpi
    """

    def __init__(self, model_path: Optional[str] = None):
        """initialize the vision trainer.

        args:
            model_path: path to mobilenetv2 tflite model. if none, uses
                        the path from simba_config.yaml.
        """
        if model_path is None:
            model_path = str(
                _project_root /
                _config["ai"]["vision_model_path"])
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = (224, 224)
        self._progress = {"current": 0, "total": 0, "phase": "idle"}
        self._load_model()

    def _load_model(self) -> None:
        """load the tflite mobilenetv2 interpreter."""
        if tflite is None:
            raise ImportError(
                "tflite runtime is required. install with: "
                "pip install ai-edge-litert"
            )
        if not os.path.isfile(self.model_path):
            fallback_path = os.path.join(os.path.dirname(self.model_path), "mobilenet_v2_1.0_224.tflite")
            if os.path.isfile(fallback_path):
                # Auto-rename the fallback model to the expected name
                try:
                    os.rename(fallback_path, self.model_path)
                    logger.info(f"Renamed {fallback_path} to {self.model_path}")
                except Exception as e:
                    logger.warning(f"Failed to rename model, using fallback path: {e}")
                    self.model_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"mobilenetv2 tflite model not found at: {self.model_path}"
                )

        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # verify expected input shape [1, 224, 224, 3]
        expected = self.input_details[0]["shape"]
        self.input_shape = (expected[1], expected[2])

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """load and preprocess an image for mobilenetv2.

        args:
            image_path: path to the image file.

        returns:
            preprocessed numpy array of shape [1, 224, 224, 3], float32,
            normalized to [-1, 1] per mobilenetv2 convention.
        """
        if Image is None:
            raise ImportError(
                "pillow is required. install with: pip install Pillow")

        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.input_shape, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)

        # mobilenetv2 normalization: [0, 255] -> [-1, 1]
        arr = (arr / 127.5) - 1.0
        arr = np.expand_dims(arr, axis=0)
        return arr

    def extract_features(self, image_path: str) -> np.ndarray:
        """extract 1280-dim feature vector from a single image.

        args:
            image_path: path to the image file.

        returns:
            numpy array of shape (1280,) — the feature vector.
        """
        preprocessed = self._preprocess_image(image_path)
        self.interpreter.set_tensor(
            self.input_details[0]["index"], preprocessed
        )
        self.interpreter.invoke()
        features = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )
        # flatten to 1d — output may be [1, 1, 1, 1280] or [1, 1280]
        return features.flatten()

    def extract_features_batch(
        self, image_paths: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """extract features from multiple images with progress tracking.

        args:
            image_paths: list of image file paths.
            show_progress: whether to show a progress bar.

        returns:
            numpy array of shape (n_images, 1280).
        """
        features_list = []
        self._progress = {
            "current": 0,
            "total": len(image_paths),
            "phase": "extracting features",
        }

        iterator = tqdm(
            image_paths,
            desc="extracting features",
            disable=not show_progress,
        )
        for i, path in enumerate(iterator):
            try:
                feat = self.extract_features(path)
                features_list.append(feat)
                if (i + 1) % 100 == 0:
                    gc.collect()
            except Exception as e:
                logger.error(f"Error extracting features from {path}: {e}")
            self._progress["current"] = i + 1

        self._progress["phase"] = "features extracted"
        result = np.array(features_list, dtype=np.float32)
        del features_list
        gc.collect()
        return result

    def train(
        self,
        data_dir: str,
        test_size: float = 0.2,
        cv_folds: int = 2,
        max_iter: int = 5000,
    ) -> Tuple[CalibratedClassifierCV, Dict[int, str], Dict]:
        """train an object classifier from a directory of class subdirectories.

        args:
            data_dir: path to directory containing one subdirectory per class,
                      each with image files (jpg, png, bmp, webp).
            test_size: fraction of data for held-out testing.
            cv_folds: number of cross-validation folds.
            max_iter: maximum svc iterations.

        returns:
            tuple of (calibrated_model, label_map, metrics_dict).

        raises:
            ValueError: if fewer than 2 classes are found.
        """
        data_path = Path(data_dir)
        if not data_path.is_dir():
            raise FileNotFoundError(f"data directory not found: {data_dir}")

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # collect image paths and labels
        image_paths = []
        labels = []
        class_names = []

        self._progress = {
            "current": 0,
            "total": 0,
            "phase": "scanning classes"}

        for class_dir in sorted(data_path.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name.lower()
            class_names.append(class_name)

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in valid_extensions:
                    image_paths.append(str(img_file))
                    labels.append(class_name)

        if len(class_names) < 2:
            raise ValueError(
                f"need at least 2 classes for training, found {
                    len(class_names)}: " f"{class_names}")

        if len(image_paths) == 0:
            raise ValueError("no images found in the data directory.")

        logger.info(
            f"found {
                len(image_paths)} images across {
                len(class_names)} classes:")
        for cn in class_names:
            count = labels.count(cn)
            if count < 2:
                raise ValueError(
                    f"class '{cn}' has only {count} images. "
                    f"need at least 2 images per class for train/test split."
                )
            logger.info(f"  {cn}: {count} images")

        # extract features
        features = self.extract_features_batch(image_paths, show_progress=True)

        # encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # build label map: {integer_index: class_name}
        label_map = {
            int(i): name.lower()
            for i, name in enumerate(label_encoder.classes_)
        }

        # split train/test
        x_train, x_test, y_train, y_test = train_test_split(
            features, encoded_labels,
            test_size=test_size,
            random_state=42,
            stratify=encoded_labels,
        )

        # clear memory
        gc.collect()

        # train linear svc with probability calibration
        self._progress["phase"] = "training svm"
        logger.info("\ntraining linear svc with probability calibration...")
        base_svc = LinearSVC(
            C=1.0,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=42,
        )
        
        # calculate max allowed folds based on smallest class count in training set
        min_class_count = min([list(y_train).count(c) for c in set(y_train)])

        if min_class_count >= 2:
            safe_cv = max(2, min(cv_folds, min_class_count))
            model = CalibratedClassifierCV(
                base_svc, cv=safe_cv, n_jobs=-1)
            model.fit(x_train, y_train)
        else:
            # Not enough samples for CV — fit base SVC first, then use prefit
            logger.warning(
                "min class count in training set is %d, using prefit calibration",
                min_class_count)
            base_svc.fit(x_train, y_train)
            model = CalibratedClassifierCV(
                base_svc, cv="prefit")
            model.fit(x_train, y_train)

        # clear memory
        del x_train, y_train
        gc.collect()

        # cross-validate
        self._progress["phase"] = "cross-validating"
        logger.info("running cross-validation...")
        
        # full dataset min class count
        min_total_class = min([list(encoded_labels).count(c) for c in set(encoded_labels)])
        safe_cv_full = max(2, min(cv_folds, min_total_class))
        
        if safe_cv_full >= 2:
            cv_scores = cross_val_score(
                model, features, encoded_labels,
                cv=safe_cv_full,
                scoring="accuracy",
                n_jobs=-1,
            )
            logger.info(f"cross-validation accuracy: {cv_scores.mean():.4f} "
                        f"(±{cv_scores.std():.4f})")
        else:
            logger.info("Not enough samples for full cross-validation. Skipping.")
            cv_scores = np.array([0.0])

        # evaluate on held-out test set
        self._progress["phase"] = "evaluating"
        y_pred = model.predict(x_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"test set accuracy: {test_accuracy:.4f}")
        logger.info("\nclassification report:")
        target_names = [label_map[i] for i in sorted(label_map.keys())]
        report = classification_report(
            y_test, y_pred, target_names=target_names)
        logger.info(report)

        n_features_dim = int(features.shape[1])

        # release remaining arrays
        del x_test, y_test, features, encoded_labels
        gc.collect()

        metrics = {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "test_accuracy": float(test_accuracy),
            "n_classes": len(class_names),
            "n_samples": len(image_paths),
            "n_features": n_features_dim,
            "class_names": list(class_names),
        }

        self._progress = {"current": 0, "total": 0, "phase": "complete"}
        return model, label_map, metrics

    def save_model(
        self,
        model: CalibratedClassifierCV,
        label_map: Dict[int, str],
        output_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """save trained svm model and label map to disk.

        args:
            model: trained calibratedclassifiercv model.
            label_map: mapping of integer labels to class names.
            output_dir: output directory. defaults to project models/ dir.

        returns:
            tuple of (model_path, label_map_path).
        """
        if output_dir is None:
            output_dir = str(_project_root / "models")
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "object_classifier.joblib")
        label_path = os.path.join(output_dir, "object_labels.json")

        joblib.dump(model, model_path)
        with open(label_path, "w") as f:
            json.dump(label_map, f, indent=2)

        logger.info(f"model saved to: {model_path}")
        logger.info(f"label map saved to: {label_path}")
        return model_path, label_path

    def evaluate(
        self,
        model: CalibratedClassifierCV,
        test_data: Tuple[np.ndarray, np.ndarray],
        label_map: Optional[Dict[int, str]] = None,
    ) -> Dict:
        """evaluate a trained model on test data.

        args:
            model: trained classifier.
            test_data: tuple of (features, labels) numpy arrays.
            label_map: optional label map for class names.

        returns:
            dict with accuracy and per-class precision/recall/f1.
        """
        features, labels = test_data
        predictions = model.predict(features)
        accuracy = accuracy_score(labels, predictions)

        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )

        per_class = {}
        unique_labels = sorted(set(labels))
        for i, label_idx in enumerate(unique_labels):
            class_name = (
                label_map.get(int(label_idx), str(label_idx))
                if label_map
                else str(label_idx)
            )
            per_class[class_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        return {
            "accuracy": float(accuracy),
            "per_class": per_class,
            "n_samples": len(labels),
        }

    @property
    def progress(self) -> Dict:
        """get current training progress for the web ui."""
        return self._progress.copy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="train simba vision model")
    parser.add_argument(
        "--data-dir",
        default=str(_project_root / "data" / "objects"),
        help="directory with class subdirectories of images",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_project_root / "models"),
        help="directory to save trained model",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="path to mobilenetv2 tflite model",
    )
    args = parser.parse_args()

    trainer = VisionTrainer(model_path=args.model_path)
    model, label_map, metrics = trainer.train(args.data_dir)
    trainer.save_model(model, label_map, args.output_dir)
    print(f"\ntraining complete. metrics: {json.dumps(metrics, indent=2)}")
