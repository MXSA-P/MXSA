# _max_cyan_ — project_mxsa
"""voice trainer — speaker verification using mfcc features + gaussian mixture model.

extracts 13 mfcc coefficients plus delta and delta-delta features from audio,
then trains a gmm to model the owner's voice. used for speaker verification
on the raspberry pi.
"""

import os
import logging
import gc
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import librosa
except ImportError:
    librosa = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    GaussianMixture = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# project root is three levels up from this file
_project_root = Path(__file__).resolve().parent.parent

# load config
_config_path = _project_root / "config" / "simba_config.yaml"
with open(_config_path, "r") as _f:
    _config = yaml.safe_load(_f)


class VoiceTrainer:
    """trains a speaker verification model using mfcc + gmm.

    workflow:
        1. collect .wav samples of the owner's voice
        2. extract 13 mfcc coefficients + delta + delta-delta (39-dim)
        3. fit a gmm (16 components) on the concatenated features
        4. at inference, score new audio against the gmm
        5. threshold the log-likelihood for verification
    """

    def __init__(
        self,
        n_mfcc: int = 13,
        n_components: int = 16,
        sample_rate: int = 16000,
        speaker_threshold: Optional[float] = None,
    ):
        """initialize the voice trainer.

        args:
            n_mfcc: number of mfcc coefficients to extract.
            n_components: number of gmm components.
            sample_rate: expected audio sample rate in hz.
            speaker_threshold: log-likelihood threshold for verification.
                              if none, uses config value.
        """
        if librosa is None:
            raise ImportError(
                "librosa is required. install with: pip install librosa"
            )
        if GaussianMixture is None:
            raise ImportError(
                "scikit-learn is required. install with: pip install scikit-learn"
            )

        self.n_mfcc = n_mfcc
        self.n_components = n_components
        self.sample_rate = sample_rate
        self.speaker_threshold = (
            speaker_threshold
            if speaker_threshold is not None
            else _config["voice"]["speaker_threshold"]
        )
        self._progress = {"current": 0, "total": 0, "phase": "idle"}

    def extract_mfcc(self, audio_path: str) -> np.ndarray:
        """extract mfcc features from an audio file.

        extracts 13 mfcc coefficients, their first-order deltas, and
        second-order deltas, producing a 39-dimensional feature per frame.

        args:
            audio_path: path to a .wav audio file.

        returns:
            numpy array of shape (n_frames, 39) — mfcc + delta + delta-delta.

        raises:
            FileNotFoundError: if audio_path does not exist.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        # load audio, resample to target rate
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return self.extract_mfcc_from_array(audio, sr=sr)

    def extract_mfcc_from_array(
        self, audio_array: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """extract mfcc features from a raw audio numpy array.

        args:
            audio_array: 1d numpy array of audio samples.
            sr: sample rate of the audio.

        returns:
            numpy array of shape (n_frames, 39).
        """
        # compute mfccs
        mfccs = librosa.feature.mfcc(
            y=audio_array,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=512,
            hop_length=160,
            n_mels=40,
        )

        # compute deltas and delta-deltas
        delta = librosa.feature.delta(mfccs, order=1)
        delta2 = librosa.feature.delta(mfccs, order=2)

        # stack: (39, n_frames) then transpose to (n_frames, 39)
        combined = np.vstack([mfccs, delta, delta2]).T
        return combined

    def train_speaker_model(
        self, audio_dir: str, show_progress: bool = True
    ) -> GaussianMixture:
        """train a gmm speaker model from a directory of .wav files.

        args:
            audio_dir: directory containing .wav files of the owner's voice.
            show_progress: whether to display progress.

        returns:
            fitted gaussianmixture model.

        raises:
            FileNotFoundError: if audio_dir does not exist.
            ValueError: if no valid audio files are found.
        """
        audio_path = Path(audio_dir)
        if not audio_path.is_dir():
            raise FileNotFoundError(f"audio directory not found: {audio_dir}")

        # find all wav files
        wav_files = sorted(
            [str(f) for f in audio_path.iterdir()
             if f.suffix.lower() == ".wav"]
        )

        if not wav_files:
            raise ValueError(f"no .wav files found in: {audio_dir}")

        logger.info(f"found {len(wav_files)} audio samples for training")

        self._progress = {
            "current": 0,
            "total": len(wav_files),
            "phase": "extracting mfcc features",
        }

        # extract features from all files
        all_features = []
        iterator = tqdm(
            wav_files,
            desc="extracting mfccs",
            disable=not show_progress)
        for i, wav_path in enumerate(iterator):
            try:
                features = self.extract_mfcc(wav_path)
                if features.shape[0] > 0:
                    all_features.append(features)
                if (i + 1) % 10 == 0:
                    gc.collect()
            except Exception as e:
                logger.error(f"Error skipping {wav_path}: {e}")
            self._progress["current"] = i + 1

        if not all_features:
            raise ValueError("no valid features extracted from audio files")

        # concatenate all frames
        combined_features = np.vstack(all_features)
        del all_features
        gc.collect()

        logger.info(f"total feature frames: {combined_features.shape[0]}")
        logger.info(f"feature dimensions: {combined_features.shape[1]}")

        if combined_features.shape[0] < 2:
            raise ValueError(
                "not enough feature frames to train a gmm. need at least 2.")

        # fit gmm
        self._progress["phase"] = "training gmm"
        logger.info(f"\nfitting gmm with {self.n_components} components...")

        # adjust n_components if we have very few frames
        n_comp = min(self.n_components, combined_features.shape[0] // 2)
        n_comp = max(n_comp, 1)

        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type="diag",
            max_iter=200,
            n_init=5,
            random_state=42,
        )
        gmm.fit(combined_features)

        # report fit quality
        avg_score = gmm.score(combined_features)
        logger.info(f"gmm average log-likelihood: {avg_score:.4f}")
        logger.info(f"gmm converged: {gmm.converged_}")

        # clear memory
        del combined_features
        gc.collect()

        self._progress = {"current": 0, "total": 0, "phase": "complete"}
        return gmm

    def save_model(
        self,
        model: "GaussianMixture",
        output_path: Optional[str] = None,
    ) -> str:
        """save the trained speaker model to disk.

        args:
            model: trained gaussianmixture model.
            output_path: path to save the model. defaults to config path.

        returns:
            path where model was saved.
        """
        if output_path is None:
            output_path = str(
                _project_root / _config["voice"]["speaker_model_path"]
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(model, output_path)
        logger.info(f"speaker model saved to: {output_path}")
        return output_path

    def test_verification(
        self,
        model: "GaussianMixture",
        test_audio: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """test speaker verification on a single audio file.

        args:
            model: trained gmm speaker model.
            test_audio: path to a .wav test file.
            threshold: verification threshold. uses instance default if none.

        returns:
            tuple of (is_owner: bool, confidence: float).
            confidence is a normalized score in [0, 1].
        """
        if threshold is None:
            threshold = self.speaker_threshold

        features = self.extract_mfcc(test_audio)
        if features.shape[0] == 0:
            return False, 0.0

        # compute average per-frame log-likelihood
        log_likelihood = model.score(features)

        # normalize to a [0, 1] confidence range using a sigmoid-like mapping
        # typical log-likelihoods for gmms on mfcc data range from -50 to 0
        # we shift and scale so that threshold maps to ~0.5
        normalized = 1.0 / (1.0 + np.exp(-(log_likelihood - threshold) * 0.5))
        confidence = float(np.clip(normalized, 0.0, 1.0))

        is_owner = log_likelihood >= threshold
        logger.info(f"log-likelihood: {log_likelihood:.4f}, "
                    f"threshold: {threshold:.4f}, "
                    f"confidence: {confidence:.4f}, "
                    f"is_owner: {is_owner}")

        return is_owner, confidence

    @property
    def progress(self) -> dict:
        """get current training progress for the web ui."""
        return self._progress.copy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="train simba speaker model")
    parser.add_argument(
        "--audio-dir",
        default=str(_project_root / "data" / "voice"),
        help="directory containing owner's .wav files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="path to save the speaker model",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="path to a test .wav file for verification",
    )
    args = parser.parse_args()

    trainer = VoiceTrainer()
    model = trainer.train_speaker_model(args.audio_dir)
    saved_path = trainer.save_model(model, args.output)

    if args.test:
        is_owner, confidence = trainer.test_verification(model, args.test)
        print(f"\nverification result: is_owner={is_owner}, "
              f"confidence={confidence:.4f}")
