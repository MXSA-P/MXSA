# _max_cyan_ — project_mxsa
"""model exporter — packages all trained models into simba_model.zip.

creates a deployable archive containing:
  - object_classifier.joblib  (svm object classifier)
  - object_labels.json        (class name mapping)
  - speaker_model.joblib      (gmm speaker verification)
  - behavior_model.joblib     (decision tree behavior)
  - simba_config.yaml         (robot configuration)
  - manifest.json             (versions, timestamps, metadata)
"""

import json
import os
import zipfile
import logging
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# project root
_project_root = Path(__file__).resolve().parent.parent

# load config
_config_path = _project_root / "config" / "simba_config.yaml"
with open(_config_path, "r") as _f:
    _config = yaml.safe_load(_f)


# files to include in the export
_model_files = {
    "object_classifier.joblib": "ai.object_classifier_path",
    "object_labels.json": "ai.object_label_map_path",
    "speaker_model.joblib": "voice.speaker_model_path",
}


def _get_model_stats(model_path: str) -> Dict:
    """extract parameter counts and metadata from a model file.

    args:
        model_path: path to a .joblib model file.

    returns:
        dict with parameter count and type information.
    """
    stats = {"exists": os.path.exists(model_path)}
    if not stats["exists"]:
        return stats

    try:
        model = joblib.load(model_path)
        stats["type"] = type(model).__name__
        stats["size_bytes"] = os.path.getsize(model_path)

        # count parameters based on model type
        if hasattr(model, "estimators_"):
            # calibratedclassifiercv wraps multiple estimators
            n_params = 0
            for est in model.estimators_:
                if hasattr(est, "estimator"):
                    inner = est.estimator
                    if hasattr(inner, "coef_"):
                        n_params += inner.coef_.size + inner.intercept_.size
            stats["n_parameters"] = int(n_params)
        elif hasattr(model, "coef_"):
            stats["n_parameters"] = int(
                model.coef_.size + model.intercept_.size
            )
        elif hasattr(model, "means_"):
            # gaussianmixture
            n_params = model.means_.size
            if model.covariances_ is not None:
                n_params += model.covariances_.size
            if model.weights_ is not None:
                n_params += model.weights_.size
            stats["n_parameters"] = int(n_params)
        elif hasattr(model, "tree_"):
            # decision tree
            stats["n_parameters"] = int(model.tree_.node_count)
            stats["tree_depth"] = int(model.get_depth())
            stats["n_leaves"] = int(model.get_n_leaves())
        else:
            stats["n_parameters"] = 0

    except Exception as e:
        logger.error(f"Error getting model stats for {model_path}: {e}")
        stats["error"] = str(e)

    return stats


class ModelExporter:
    """packages trained models into a deployable zip archive.

    creates simba_model.zip containing all models, config, and a manifest
    with version info and parameter counts. the zip can be transferred
    to the raspberry pi for deployment.
    """

    def __init__(self):
        """initialize the model exporter."""
        self._progress = {"current": 0, "total": 0, "phase": "idle"}

    def export(
        self,
        output_path: Optional[str] = None,
        models_dir: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> str:
        """export all models to a zip archive.

        args:
            output_path: path for the output .zip file. defaults to
                        models/simba_model.zip.
            models_dir: directory containing trained models. defaults to
                       project models/ directory.
            config_path: path to simba_config.yaml. defaults to project config.

        returns:
            path to the created zip file.

        raises:
            FileNotFoundError: if required model files are missing.
        """
        if models_dir is None:
            models_dir = str(_project_root / "models")
        if config_path is None:
            config_path = str(_config_path)
        if output_path is None:
            output_path = os.path.join(models_dir, "simba_model.zip")

        models_path = Path(models_dir)
        self._progress = {"current": 0, "total": 5, "phase": "checking files"}

        # verify required files exist
        required_files = {
            "object_classifier.joblib": models_path /
            "object_classifier.joblib",
            "object_labels.json": models_path /
            "object_labels.json",
            "speaker_model.joblib": models_path /
            "speaker_model.joblib",
            "behavior_model.joblib": models_path /
            "behavior_model.joblib",
        }

        missing = []
        for name, path in required_files.items():
            if not path.exists():
                missing.append(name)

        if missing:
            raise FileNotFoundError(
                f"missing model files: {', '.join(missing)}. "
                f"train all models before exporting."
            )

        self._progress = {
            "current": 1,
            "total": 5,
            "phase": "collecting stats"}

        # collect model stats
        model_stats = {}
        for name, path in required_files.items():
            model_stats[name] = _get_model_stats(str(path))

        self._progress = {
            "current": 2,
            "total": 5,
            "phase": "building manifest"}

        # build manifest
        total_params = sum(
            s.get("n_parameters", 0) for s in model_stats.values()
        )
        manifest = {
            "_max_cyan_": "project_mxsa",
            "watermark": _config["robot"]["watermark"],
            "robot_name": _config["robot"]["name"],
            "version": _config["robot"]["version"],
            "export_timestamp": datetime.now().isoformat(),
            "exported_by": "simba model exporter",
            "models": model_stats,
            "total_parameters": total_params,
            "config_included": True,
            "files": list(required_files.keys()) + [
                "simba_config.yaml",
                "manifest.json",
            ],
        }

        self._progress = {
            "current": 3,
            "total": 5,
            "phase": "creating archive"}

        # create zip archive
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # add model files
            for name, path in required_files.items():
                zf.write(str(path), name)

            # add config
            if os.path.exists(config_path):
                zf.write(config_path, "simba_config.yaml")

            # add manifest
            manifest_json = json.dumps(manifest, indent=2)
            zf.writestr("manifest.json", manifest_json)

        self._progress = {"current": 4, "total": 5, "phase": "verifying"}

        # verify archive
        zip_size = os.path.getsize(output_path)
        with zipfile.ZipFile(output_path, "r") as zf:
            file_list = zf.namelist()
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"corrupt file in archive: {bad}")

        self._progress = {"current": 5, "total": 5, "phase": "complete"}

        logger.info(f"\nexport complete: {output_path}")
        logger.info(f"archive size: {zip_size / 1024:.1f} kb")
        logger.info(f"files included: {len(file_list)}")
        for f in file_list:
            logger.info(f"  - {f}")
        logger.info(f"total model parameters: {total_params}")

        gc.collect()
        return output_path

    def get_manifest(self, zip_path: str) -> Dict:
        """read the manifest from an exported zip.

        args:
            zip_path: path to simba_model.zip.

        returns:
            manifest dict.
        """
        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest_data = zf.read("manifest.json")
            return json.loads(manifest_data)

    @property
    def progress(self) -> dict:
        """get current export progress for the web ui."""
        return self._progress.copy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="export simba models")
    parser.add_argument(
        "--output",
        default=None,
        help="output zip file path",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="directory containing trained models",
    )
    args = parser.parse_args()

    exporter = ModelExporter()
    zip_path = exporter.export(
        output_path=args.output,
        models_dir=args.models_dir,
    )
    print("\nmanifest:")
    manifest = exporter.get_manifest(zip_path)
    print(json.dumps(manifest, indent=2))
