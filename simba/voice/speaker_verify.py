# _max_cyan_ — project_mxsa
"""Speaker verification using MFCC features and Gaussian Mixture Model.

Extracts 13 MFCC coefficients plus delta features from audio chunks
using librosa, then uses a scikit-learn GaussianMixture model to
determine whether the speaker is the enrolled owner.
"""

from collections import deque
import os
import threading
from typing import Any, Deque, Dict, List, Optional, Tuple

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
    """Speaker verification using MFCC + GMM.

    Extracts MFCC features and their deltas from audio, then uses
    a trained Gaussian Mixture Model to verify whether the speaker
    is the enrolled owner. The model is persisted to disk using joblib.

    Attributes:
        threshold: Minimum log-likelihood score to accept as owner.
        sample_rate: Expected audio sample rate in Hz.
        model: Trained GaussianMixture instance, or None.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the speaker verifier.

        Args:
            config: Full Simba configuration dict loaded from YAML.
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

        self.model: Optional[Any] = None
        self._enrolled: bool = False
        self._last_confidence: float = 0.0
        self._baseline_score: float = 0.0

        # continuous verification and confidence history tracking
        self._history_max_len: int = 20
        self._confidence_history: Deque[float] = deque(maxlen=self._history_max_len)

        # load existing model if available
        self._load_model()

        log_event("voice", "speaker verifier initialized", {
            "enrolled": self._enrolled,
            "threshold": self.threshold,
            "librosa_available": _has_librosa,
            "sklearn_available": _has_sklearn,
        })

    def _load_model(self) -> None:
        """Load a previously trained speaker model from disk."""
        if not _has_joblib or not _has_sklearn:
            logger.warning("joblib or sklearn not available — "
                           "speaker verification disabled")
            return

        if os.path.isfile(self._model_path):
            try:
                saved = joblib.load(self._model_path)
                if isinstance(saved, dict):
                    self.model = saved.get("model")
                    self._baseline_score = float(saved.get("baseline_score", 0.0))
                else:
                    # legacy format (raw model)
                    self.model = saved
                    self._baseline_score = 0.0
                self._enrolled = self.model is not None
                logger.info("speaker model loaded from %s", self._model_path)
            except (EOFError, ValueError, OSError) as exc:
                logger.error("corrupted speaker model file detected (%s), resetting: %s", self._model_path, exc)
                self._enrolled = False
                self.model = None
                self._baseline_score = 0.0
                try:
                    os.remove(self._model_path)
                except OSError:
                    pass
            except Exception as exc:
                logger.error("failed to load speaker model: %s", exc)
                self._enrolled = False
                self.model = None
                self._baseline_score = 0.0
        else:
            logger.info("no speaker model found at %s — not enrolled",
                        self._model_path)

    def _extract_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract MFCC features and deltas from an audio chunk.

        Extracts 13 MFCC coefficients and their first-order deltas,
        producing a 26-dimensional feature vector per frame.

        Args:
            audio: 1D numpy array of audio samples (float32, mono).

        Returns:
            2D numpy array of shape (n_frames, 26) or None on failure.
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
        """Verify whether the given audio belongs to the enrolled owner.

        Args:
            audio_chunk: 1D numpy array of audio samples (float32 or int16).

        Returns:
            Tuple of (is_owner, confidence). is_owner is True if the
            speaker matches the enrolled model above the threshold.
            confidence is a float between 0.0 and 1.0.
        """
        with self._lock:
            if not self._enrolled or self.model is None:
                logger.debug("no enrolled speaker — open mic fallback active")
                self._last_confidence = 1.0
                self._confidence_history.append(1.0)
                return (True, 1.0)

            # convert int16 to float32 if needed
            if audio_chunk.dtype == np.int16:
                audio_float = audio_chunk.astype(np.float32) / 32768.0
            else:
                audio_float = audio_chunk.astype(np.float32)

            features = self._extract_features(audio_float)
            if features is None:
                self._last_confidence = 0.0
                self._confidence_history.append(0.0)
                return (False, 0.0)

            try:
                # compute per-frame log-likelihood and average
                score = self.model.score(features)

                # normalize score to 0-1 range using baseline
                # positive difference from baseline means more likely owner
                if self._baseline_score != 0:
                    # shift the sigmoid so a score equal to baseline gives ~0.88 confidence
                    normalized = 1.0 / (1.0 + np.exp(
                        -(score - self._baseline_score + 2.0)
                    ))
                else:
                    # without baseline, use sigmoid of raw score
                    normalized = 1.0 / (1.0 + np.exp(-score / 10.0))

                confidence = float(np.clip(normalized, 0.0, 1.0))
                self._last_confidence = confidence

                # history tracking
                self._confidence_history.append(confidence)

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
                self._confidence_history.append(0.0)
                return (False, 0.0)

    def enroll(self, audio_samples: List[np.ndarray]) -> bool:
        """Enroll a new speaker by training the GMM on audio samples.

        Args:
            audio_samples: List of 1D numpy arrays, each containing
                           a speech utterance from the owner.

        Returns:
            True if enrollment succeeded, False otherwise.
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
                self._confidence_history.clear()

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
        """Check if a speaker has been enrolled.

        Returns:
            True if a speaker model is loaded/trained.
        """
        return self._enrolled

    def get_confidence(self) -> float:
        """Get the confidence score from the last verification.

        Returns:
            Confidence float between 0.0 and 1.0.
        """
        return self._last_confidence

    def get_confidence_history(self) -> List[float]:
        """Get the recent history of confidence scores.

        Returns:
            List of recent confidence scores.
        """
        with self._lock:
            return list(self._confidence_history)

    def continuous_verify(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """Perform continuous verification using historical average.

        Args:
            audio_chunk: 1D numpy array of audio samples.

        Returns:
            (is_owner, avg_confidence) tuple based on smoothed history.
        """
        self.verify(audio_chunk)
        with self._lock:
            if not self._enrolled or self.model is None:
                return (True, 1.0)
            if not self._confidence_history:
                return (False, 0.0)
            avg_confidence = sum(self._confidence_history) / \
                max(1, len(self._confidence_history))
            is_owner = avg_confidence >= self.threshold
            return (is_owner, avg_confidence)

    def __repr__(self) -> str:
        """Return a string representation of the SpeakerVerifier.

        Returns:
            String representation showing enrollment status, threshold, and last confidence.
        """
        return (
            f"SpeakerVerifier(enrolled={self._enrolled}, "
            f"threshold={self.threshold}, "
            f"last_confidence={self._last_confidence:.3f})"
        )
