# _max_cyan_ — project_mxsa
"""speaker verification using mfcc features and gaussian mixture model.

extracts 13 mfcc coefficients plus delta features from audio chunks
using librosa, then uses a scikit-learn gaussianmixture model to
determine whether the speaker is the enrolled owner.
"""

import os
import threading
from typing import Optional, Tuple

import numpy as np

try:
    import librosa
    _has_librosa = True
except ImportError:
    librosa = None
    _has_librosa = False

try:
    from sklearn.mixture import GaussianMixture
    _has_sklearn = True
except ImportError:
    GaussianMixture = None
    _has_sklearn = False

try:
    import joblib
    _has_joblib = True
except ImportError:
    joblib = None
    _has_joblib = False

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.voice.speaker_verify")

# number of mfcc coefficients to extract
_n_mfcc = 13

# number of gaussian components for the gmm
_n_components = 16

# minimum audio length in samples (0.5 seconds at 16khz)
_min_audio_length = 8000


class SpeakerVerifier:
    """speaker verification using mfcc + gmm.

    extracts mfcc features and their deltas from audio, then uses
    a trained gaussian mixture model to verify whether the speaker
    is the enrolled owner. the model is persisted to disk using joblib.

    attributes:
        threshold: minimum log-likelihood score to accept as owner.
        sample_rate: expected audio sample rate in hz.
        model: trained gaussianmixture instance, or none.
    """

    def __init__(self, config: dict) -> None:
        """initialize the speaker verifier.

        args:
            config: full simba configuration dict loaded from yaml.
        """
        self._lock = threading.Lock()

        voice_cfg = config.get("voice", {})
        self.threshold: float = voice_cfg.get("speaker_threshold", 0.65)
        self.sample_rate: int = voice_cfg.get("sample_rate", 16000)

        # resolve model path relative to project root
        model_path = voice_cfg.get(
            "speaker_model_path",
            "models/speaker_model.joblib")
        if not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ))
            model_path = os.path.join(project_root, model_path)
        self._model_path: str = model_path

        self.model: Optional[object] = None
        self._enrolled: bool = False
        self._last_confidence: float = 0.0
        self._baseline_score: float = 0.0

        # continuous verification and confidence history tracking
        self._confidence_history: list[float] = []
        self._history_max_len: int = 20

        # load existing model if available
        self._load_model()

        log_event("voice", "speaker verifier initialized", {
            "enrolled": self._enrolled,
            "threshold": self.threshold,
            "librosa_available": _has_librosa,
            "sklearn_available": _has_sklearn,
        })

    def _load_model(self) -> None:
        """load a previously trained speaker model from disk."""
        if not _has_joblib or not _has_sklearn:
            logger.warning("joblib or sklearn not available — "
                           "speaker verification disabled")
            return

        if os.path.isfile(self._model_path):
            try:
                saved = joblib.load(self._model_path)
                if isinstance(saved, dict):
                    self.model = saved.get("model")
                    self._baseline_score = saved.get("baseline_score", 0.0)
                else:
                    # legacy format (raw model)
                    self.model = saved
                    self._baseline_score = 0.0
                self._enrolled = self.model is not None
                logger.info("speaker model loaded from %s", self._model_path)
            except Exception as exc:
                logger.error("failed to load speaker model: %s", exc)
                self._enrolled = False
        else:
            logger.info("no speaker model found at %s — not enrolled",
                        self._model_path)

    def _extract_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """extract mfcc features and deltas from an audio chunk.

        extracts 13 mfcc coefficients and their first-order deltas,
        producing a 26-dimensional feature vector per frame.

        args:
            audio: 1d numpy array of audio samples (float32, mono).

        returns:
            2d numpy array of shape (n_frames, 26) or none on failure.
        """
        if not _has_librosa:
            logger.error("librosa not available — cannot extract features")
            return None

        if audio is None or len(audio) < _min_audio_length:
            logger.debug("audio too short for feature extraction "
                         "(%d < %d samples)", len(audio), _min_audio_length)
            return None

        try:
            # ensure float32 mono
            audio_float = audio.astype(np.float32).flatten()

            # normalize amplitude
            peak = np.max(np.abs(audio_float))
            if peak > 0:
                audio_float = audio_float / peak

            # extract mfcc coefficients
            mfccs = librosa.feature.mfcc(
                y=audio_float,
                sr=self.sample_rate,
                n_mfcc=_n_mfcc,
                n_fft=512,
                hop_length=160,
                n_mels=40,
            )

            # compute first-order deltas
            deltas = librosa.feature.delta(mfccs)

            # stack mfccs and deltas: shape (26, n_frames) -> (n_frames, 26)
            features = np.vstack([mfccs, deltas]).T

            return features

        except Exception as exc:
            logger.error("feature extraction failed: %s", exc)
            return None

    def verify(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """verify whether the given audio belongs to the enrolled owner.

        args:
            audio_chunk: 1d numpy array of audio samples (float32 or int16).

        returns:
            tuple of (is_owner, confidence). is_owner is true if the
            speaker matches the enrolled model above the threshold.
            confidence is a float between 0.0 and 1.0.
        """
        with self._lock:
            if not self._enrolled or self.model is None:
                logger.debug("no enrolled speaker — verification skipped")
                self._last_confidence = 0.0
                return (False, 0.0)

            # convert int16 to float32 if needed
            if audio_chunk.dtype == np.int16:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_float = audio_chunk.astype(np.float32)

            features = self._extract_features(audio_float)
            if features is None:
                self._last_confidence = 0.0
                return (False, 0.0)

            try:
                # compute per-frame log-likelihood and average
                score = self.model.score(features)

                # normalize score to 0-1 range using baseline
                # positive difference from baseline means more likely owner
                if self._baseline_score != 0:
                    normalized = 1.0 / (1.0 + np.exp(
                        -(score - self._baseline_score)
                    ))
                else:
                    # without baseline, use sigmoid of raw score
                    normalized = 1.0 / (1.0 + np.exp(-score / 10.0))

                confidence = float(np.clip(normalized, 0.0, 1.0))
                self._last_confidence = confidence

                # history tracking
                self._confidence_history.append(confidence)
                if len(self._confidence_history) > self._history_max_len:
                    self._confidence_history.pop(0)

                is_owner = confidence >= self.threshold

                logger.debug("speaker verify: score=%.3f, confidence=%.3f, "
                             "is_owner=%s", score, confidence, is_owner)

                log_event("voice", "speaker verification", {
                    "is_owner": is_owner,
                    "confidence": round(confidence, 3),
                    "raw_score": round(score, 3),
                })

                return (is_owner, confidence)

            except Exception as exc:
                logger.error("speaker verification failed: %s", exc)
                self._last_confidence = 0.0
                return (False, 0.0)

    def enroll(self, audio_samples: list) -> bool:
        """enroll a new speaker by training the gmm on audio samples.

        args:
            audio_samples: list of 1d numpy arrays, each containing
                          a speech utterance from the owner.

        returns:
            true if enrollment succeeded, false otherwise.
        """
        if not _has_sklearn or not _has_librosa or not _has_joblib:
            logger.error("required libraries not available for enrollment")
            return False

        if not audio_samples or len(audio_samples) < 3:
            logger.error("need at least 3 audio samples for enrollment")
            return False

        try:
            all_features = []
            for audio in audio_samples:
                if isinstance(audio, np.ndarray):
                    if audio.dtype == np.int16:
                        audio = audio.astype(np.float32) / 32768.0
                    features = self._extract_features(audio.astype(np.float32))
                    if features is not None:
                        all_features.append(features)

            if len(all_features) < 3:
                logger.error("could not extract features from enough samples")
                return False

            # concatenate all feature frames
            combined = np.vstack(all_features)

            logger.info("training speaker model on %d frames from %d samples",
                        combined.shape[0], len(all_features))

            # train gmm
            n_comps = min(_n_components, combined.shape[0] // 2)
            gmm = GaussianMixture(
                n_components=max(2, n_comps),
                covariance_type="diag",
                max_iter=200,
                n_init=3,
                random_state=42,
            )
            gmm.fit(combined)

            # compute baseline score (average score on training data)
            baseline = float(gmm.score(combined))

            # save model
            with self._lock:
                self.model = gmm
                self._baseline_score = baseline
                self._enrolled = True

            # persist to disk
            os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
            joblib.dump(
                {"model": gmm, "baseline_score": baseline},
                self._model_path,
            )

            logger.info("speaker enrolled successfully (baseline=%.3f, "
                        "saved to %s)", baseline, self._model_path)
            log_event("voice", "speaker enrolled", {
                "n_samples": len(all_features),
                "n_frames": combined.shape[0],
                "baseline_score": round(baseline, 3),
            })
            return True

        except Exception as exc:
            logger.error("enrollment failed: %s", exc)
            return False

    def is_enrolled(self) -> bool:
        """check if a speaker has been enrolled.

        returns:
            true if a speaker model is loaded/trained.
        """
        return self._enrolled

    def get_confidence(self) -> float:
        """get the confidence score from the last verification.

        returns:
            confidence float between 0.0 and 1.0.
        """
        return self._last_confidence

    def get_confidence_history(self) -> list[float]:
        """get the recent history of confidence scores."""
        with self._lock:
            return list(self._confidence_history)

    def continuous_verify(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """perform continuous verification using historical average.

        args:
            audio_chunk: 1d numpy array of audio samples.

        returns:
            (is_owner, avg_confidence) tuple based on smoothed history.
        """
        self.verify(audio_chunk)
        with self._lock:
            if not self._confidence_history:
                return (False, 0.0)
            avg_confidence = sum(self._confidence_history) / \
                len(self._confidence_history)
            is_owner = avg_confidence >= self.threshold
            return (is_owner, avg_confidence)

    def __repr__(self) -> str:
        return (
            f"SpeakerVerifier(enrolled={self._enrolled}, "
            f"threshold={self.threshold}, "
            f"last_confidence={self._last_confidence:.3f})"
        )
