# _max_cyan_ — project_mxsa
"""always-on voice listener using vosk for continuous speech recognition.

captures audio from the inmp441 i2s microphone via sounddevice,
feeds chunks to the vosk recognizer, and dispatches recognized text
to registered callbacks. no wake word — always active.
"""

import collections
import json
import os
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

try:
    import sounddevice as sd
    _has_sounddevice = True
except (ImportError, OSError):
    sd = None
    _has_sounddevice = False

try:
    from vosk import Model, KaldiRecognizer
    _has_vosk = True
except ImportError:
    Model = None
    KaldiRecognizer = None
    _has_vosk = False

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.voice.listener")


class VoiceListener:
    """continuous speech recognition using vosk.

    listens on the i2s microphone via sounddevice, feeds audio chunks
    to the vosk recognizer, and notifies registered callbacks when
    text is recognized. all output text is lowercase.

    attributes:
        sample_rate: audio sample rate in hz.
        channels: number of audio channels (mono).
        chunk_size: number of samples per audio chunk.
        model: loaded vosk model instance.
        recognizer: vosk kaldi recognizer instance.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """initialize the voice listener.

        args:
            config: full simba configuration dict loaded from yaml.
        """
        self._lock: threading.Lock = threading.Lock()
        self._callbacks: List[Callable[[str], None]] = []
        self._last_text: str = ""
        self._listening: bool = False
        self._stream: Optional[object] = None
        self._audio_queue: Deque[bytes] = collections.deque(maxlen=100)
        self._process_thread: Optional[threading.Thread] = None

        # energy monitoring and dynamic silence threshold
        self._current_energy: float = 0.0
        self._ambient_energy: float = 0.01
        self._energy_threshold: float = 0.05
        self._is_speaking: bool = False
        self._silence_frames: int = 0

        voice_cfg = config.get("voice", {})
        self.sample_rate: int = voice_cfg.get("sample_rate", 16000)
        self.channels: int = voice_cfg.get("channels", 1)
        self.chunk_size: int = voice_cfg.get("chunk_size", 4000)

        # resolve vosk model path relative to project root
        model_path = voice_cfg.get(
            "vosk_model_path",
            "models/vosk-model-small-en-us-0.15")
        if not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ))
            model_path = os.path.join(project_root, model_path)

        self.model = None
        self.recognizer = None

        if _has_vosk and os.path.isdir(model_path):
            try:
                self.model = Model(model_path)
                self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
                logger.info("vosk model loaded from %s", model_path)
            except Exception as exc:
                logger.error("failed to load vosk model: %s", exc)
        else:
            if not _has_vosk:
                logger.warning("vosk not installed — voice listener disabled")
            elif not os.path.isdir(model_path):
                logger.warning("vosk model not found at %s", model_path)

        log_event("voice", "voice listener initialized", {
            "sample_rate": self.sample_rate,
            "model_loaded": self.model is not None,
        })

    def _audio_callback(
        self, indata: np.ndarray, frames: int,
        time_info: Dict[str, Any], status: Any
    ) -> None:
        """sounddevice input stream callback — process incoming audio.

        args:
            indata: numpy array of audio data.
            frames: number of frames in this chunk.
            time_info: timing information from portaudio.
            status: status flags from portaudio.
        """
        if status:
            logger.warning("portaudio status: %s", status)
            if hasattr(status, 'input_overflow') and status.input_overflow:
                logger.warning(
                    "buffer overrun detected (input overflow) - "
                    "clearing audio queue"
                )
                self._audio_queue.clear()

        # Guard against invalid frame sizes and bad input types
        if self.recognizer is None:
            return

        if not isinstance(frames, int) or frames <= 0:
            return

        if not isinstance(indata, np.ndarray) or indata.size == 0:
            return

        # Guard against size mismatch
        if indata.shape[0] != frames:
            logger.debug(
                "invalid frame size or mismatch: frames=%s, indata.shape=%s",
                frames, indata.shape
            )
            return

        # convert float32 numpy array to int16 bytes for vosk
        try:
            if indata.ndim != 2 or indata.shape[1] == 0:
                return
            if np.isnan(indata).any() or np.isinf(indata).any():
                return

            # Average channels to support both L/R INMP441 wiring
            if indata.ndim == 2 and indata.shape[1] >= 2:
                audio_data = np.mean(indata, axis=1)
            elif indata.ndim == 2:
                audio_data = indata[:, 0]
            else:
                audio_data = indata

            # calculate rms energy
            energy = float(np.sqrt(np.mean(audio_data ** 2)))
            if np.isnan(energy) or np.isinf(energy):
                return

            self._current_energy = energy

            # update ambient energy (dynamic threshold)
            # adapt slowly to background noise
            self._ambient_energy = (
                self._ambient_energy * 0.995 + energy * 0.005
            )
            self._energy_threshold = max(0.01, self._ambient_energy * 2.5)

            if energy > self._energy_threshold:
                self._is_speaking = True
                self._silence_frames = 0
            else:
                self._silence_frames += 1
                # about 1 sec of silence at 4000 chunks/16khz
                if self._silence_frames > 5:
                    self._is_speaking = False

            audio_data_clipped = np.clip(audio_data * 32767, -32768, 32767)
            audio_bytes = audio_data_clipped.astype(np.int16).tobytes()
            if len(self._audio_queue) == self._audio_queue.maxlen:
                logger.warning(
                    "audio queue full, buffer overrun. dropping oldest frame"
                )
            self._audio_queue.append(audio_bytes)
        except Exception as exc:
            logger.error("audio processing error: %s", exc)
            # Don't raise CallbackAbort — a single corrupt chunk
            # shouldn't permanently kill the audio stream
            return

    def _process_audio(self) -> None:
        """background thread for processing audio queue with vosk."""
        while self._listening:
            try:
                audio_bytes = self._audio_queue.popleft()
            except IndexError:
                time.sleep(0.01)
                continue

            # Guard against unexpected byte lengths for 16-bit PCM
            # (must be a multiple of 2)
            if (not isinstance(audio_bytes, bytes) or len(audio_bytes) == 0 or
                    len(audio_bytes) % 2 != 0):
                continue

            try:
                if self.recognizer.AcceptWaveform(audio_bytes):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip().lower()
                    if text:
                        self._dispatch_text(text)
                else:
                    # check partial results for real-time feedback
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip().lower()
                    if partial_text:
                        logger.debug("partial: %s", partial_text)
            except Exception as exc:
                logger.error("recognizer error: %s", exc)

    def _dispatch_text(self, text: str) -> None:
        """store recognized text and notify all registered callbacks.

        args:
            text: recognized text string (already lowercase).
        """
        with self._lock:
            self._last_text = text

        logger.info("recognized: '%s'", text)
        log_event("voice", "speech recognized", {"text": text})

        for callback in list(self._callbacks):
            def _run(cb, t):
                try:
                    cb(t)
                except Exception as exc:
                    logger.error("callback error for text '%s': %s", t, exc)

            threading.Thread(
                target=_run, args=(callback, text), daemon=True
            ).start()

    def start(self) -> bool:
        """begin continuous listening on the i2s microphone.

        returns:
            true if listening started successfully, false otherwise.
        """
        if self._listening:
            logger.warning("listener already running")
            return True

        if not _has_sounddevice:
            logger.error("sounddevice not available — cannot start listener")
            return False

        if self.recognizer is None:
            logger.error("vosk recognizer not loaded — cannot start listener")
            return False

        try:
            device_id = None
            is_i2s = False
            try:
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    name = dev['name'].lower()
                    if dev['max_input_channels'] > 0 and (
                            'i2s' in name or 'inmp441' in name or
                            'snd' in name or 'mic' in name):
                        device_id = i
                        logger.info(
                            "Found potential I2S microphone "
                            "at device index %d: %s",
                            i, dev['name']
                        )
                        if ('i2s' in name or 'snd' in name or
                                'voicehat' in name):
                            is_i2s = True
                            self.channels = 2
                            self.sample_rate = 48000
                            from vosk import KaldiRecognizer
                            self.recognizer = KaldiRecognizer(
                                self.model, self.sample_rate
                            )
                        break
            except Exception as e:
                logger.warning("Failed to query devices for I2S mic: %s", e)

            try:
                self._stream = sd.InputStream(
                    device=device_id,
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    blocksize=self.chunk_size,
                    callback=self._audio_callback,
                )
            except Exception as e_default:
                if not is_i2s:
                    logger.warning(
                        "failed to open audio stream (device=%s): %s",
                        device_id, e_default
                    )
                    if "sample rate" in str(e_default).lower():
                        logger.warning(
                            "Falling back to 48000Hz for compatibility."
                        )
                        self.sample_rate = 48000
                        from vosk import KaldiRecognizer
                        self.recognizer = KaldiRecognizer(
                            self.model, self.sample_rate
                        )

                try:
                    self._stream = sd.InputStream(
                        device=device_id,
                        samplerate=self.sample_rate,
                        channels=2,  # Fallback to stereo for generic I2S mics
                        dtype="float32",
                        blocksize=self.chunk_size,
                        callback=self._audio_callback,
                    )
                except Exception as e_fallback:
                    if ("device -1" in str(e_fallback) or
                            "Error querying device" in str(e_fallback)):
                        logger.warning(
                            "No microphone detected (device -1). "
                            "Voice listener disabled."
                        )
                    else:
                        logger.error(
                            "failed to open 2-channel audio fallback: %s",
                            e_fallback
                        )
                    return False

            self._stream.start()
            self._listening = True
            self._process_thread = threading.Thread(
                target=self._process_audio, daemon=True
            )
            self._process_thread.start()
            logger.info("voice listener started (rate=%d, chunk=%d)",
                        self.sample_rate, self.chunk_size)
            log_event("voice", "listener started")
            return True
        except Exception as exc:
            logger.error("failed to start audio stream: %s", exc)
            self._listening = False
            if getattr(self, '_stream', None) is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            return False

    def stop(self) -> None:
        """stop listening and release the audio stream."""
        was_listening = self._listening
        self._listening = False

        try:
            if getattr(self, '_stream', None) is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception as exc:
            logger.error("error stopping audio stream: %s", exc)
        finally:
            if self._process_thread and self._process_thread.is_alive():
                if threading.current_thread() != self._process_thread:
                    self._process_thread.join(timeout=2)
            if was_listening:
                logger.info("voice listener stopped")
                log_event("voice", "listener stopped")

    def on_text(self, callback: Callable[[str], None]) -> None:
        """register a callback to receive recognized text.

        the callback will be invoked with a single string argument
        containing the recognized text in lowercase.

        args:
            callback: function that accepts a string argument.
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug("registered text callback: %s", callback.__name__)

    def remove_callback(self, callback: Callable[[str], None]) -> None:
        """remove a registered callback.

        args:
            callback: function that accepts a string argument.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug("removed text callback: %s", callback.__name__)

    def get_last_text(self) -> str:
        """return the most recently recognized text.

        returns:
            the last recognized text string, or empty string if none.
        """
        with self._lock:
            return self._last_text

    def get_energy(self) -> float:
        """return the current rms energy of the audio stream.

        returns:
            the current rms energy as a float.
        """
        return self._current_energy

    def is_speaking(self) -> bool:
        """check if someone is currently speaking (above dynamic threshold).

        returns:
            true if speaking, false otherwise.
        """
        return self._is_speaking

    def is_listening(self) -> bool:
        """check if the listener is currently active.

        returns:
            true if listening, false otherwise.
        """
        return self._listening

    def __del__(self) -> None:
        """ensure cleanup on garbage collection."""
        self.stop()

    def __repr__(self) -> str:
        """return string representation of the voice listener."""
        return (
            f"VoiceListener(listening={self._listening}, "
            f"rate={self.sample_rate}, model_loaded={self.model is not None})"
        )
