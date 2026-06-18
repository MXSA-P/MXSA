# _max_cyan_ — project_mxsa
"""emotional state machine for simba robot — v2.

manages simba's emotional state with:
- primary + secondary emotion blending
- long-term mood baseline that shifts slowly
- personality traits (curiosity, playfulness, loyalty, sensitivity)
- event-driven updates with text responses
- motor behavior mapping per emotion
- natural decay toward neutral (higher intensity decays slower)
- color codes for dashboard visualization

thread-safe for concurrent access from voice, vision, and brain modules.
"""

import random
import threading
import time
from typing import Any, Dict, List, Tuple

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.core.emotions")

# all valid emotions
_valid_emotions = frozenset({
    "happy", "sad", "curious", "excited", "proud",
    "love", "sleepy", "angry", "neutral",
    "frustrated", "surprised", "relaxed", "focused", "anxious", "bored",
    "scared", "confused", "lonely", "determined", "guilty"
})

# emoji mapping for each emotion
_emotion_emojis: Dict[str, str] = {
    "happy": "😊",
    "sad": "😢",
    "curious": "🤔",
    "excited": "🤩",
    "proud": "😤",
    "love": "❤️",
    "sleepy": "😴",
    "angry": "😠",
    "neutral": "😐",
    "frustrated": "😫",
    "surprised": "😲",
    "relaxed": "😌",
    "focused": "🧐",
    "anxious": "😰",
    "bored": "🥱",
    "scared": "😨",
    "confused": "🥴",
    "lonely": "🥺",
    "determined": "🔥",
    "guilty": "😣",
}

# color hex codes for each emotion (for dashboard glow effects)
_emotion_colors: Dict[str, str] = {
    "happy": "#22c55e",
    "sad": "#3b82f6",
    "curious": "#06b6d4",
    "excited": "#f59e0b",
    "proud": "#a855f7",
    "love": "#ec4899",
    "sleepy": "#6b7280",
    "angry": "#ef4444",
    "neutral": "#9ca3af",
    "frustrated": "#f43f5e",
    "surprised": "#fbbf24",
    "relaxed": "#10b981",
    "focused": "#8b5cf6",
    "anxious": "#64748b",
    "bored": "#d1d5db",
    "scared": "#475569",
    "confused": "#d946ef",
    "lonely": "#6366f1",
    "determined": "#ff3300",
    "guilty": "#84cc16",
}

# event -> emotion mapping with default intensities
_event_emotion_map: Dict[str, Tuple[str, float]] = {
    "object_found": ("proud", 0.8),
    "object_not_found": ("sad", 0.6),
    "praised": ("happy", 1.0),
    "scolded": ("sad", 0.7),
    "greeted": ("happy", 0.8),
    "loved": ("love", 1.0),
    "playing": ("excited", 0.9),
    "idle_long": ("sleepy", 0.5),
    "task_complete": ("proud", 0.9),
    "low_battery": ("sleepy", 0.7),
    "tapped": ("curious", 0.7),
    "shaken": ("excited", 0.8),
    "fell": ("angry", 0.6),
    "new_object": ("curious", 0.8),
    "owner_nearby": ("happy", 0.7),
    "alone_long": ("sad", 0.4),
    "dancing": ("excited", 1.0),
    "scared_by_noise": ("scared", 0.9),
    "made_mistake": ("guilty", 0.7),
    "lost": ("confused", 0.8),
    "ignored": ("lonely", 0.6),
    "mission_start": ("determined", 0.9),
    "returning_home": ("determined", 0.8),
    "arrived_home": ("happy", 0.9),
}

# event -> text responses (multiple per event for variety)
_event_responses: Dict[str, List[str]] = {
    "object_found": [
        "found it! i'm so proud of myself! 😤",
        "there it is! mission accomplished! 🎯",
        "got it! was that fast enough? 😊",
    ],
    "object_not_found": [
        "i couldn't find it... sorry 😢",
        "it's not here... i'll keep looking",
        "no luck this time 😔",
    ],
    "praised": [
        "thank you! that means a lot! 😊",
        "yay! i'm a good boy! 🐾",
        "i'll keep doing my best! ✨",
        "*happy wiggle* 🎉",
    ],
    "scolded": [
        "i'm sorry... i'll try harder 😢",
        "oh no... what did i do wrong? 😔",
        "i'll do better next time...",
    ],
    "greeted": [
        "hello there! nice to see you! 👋",
        "hi hi hi! i missed you! 😊",
        "hey! what can i do for you? 🐾",
    ],
    "loved": [
        "i love you too!! ❤️❤️❤️",
        "you make me so happy! 🥰",
        "*excited spinning* i love you!! ❤️🦁",
    ],
    "playing": [
        "yay! let's play! 🎮",
        "this is so fun! 🎉",
        "play play play! wheee! 😄",
    ],
    "idle_long": [
        "getting sleepy... *yawn* 😴",
        "is there anything to do? 💤",
        "zzzz... 😴",
    ],
    "task_complete": [
        "all done! nailed it! 😤✨",
        "mission complete! what's next? 🎯",
        "finished! i'm on fire today! 🔥",
    ],
    "low_battery": [
        "i'm getting tired... need to charge 🔋",
        "battery low... sleepy time soon 😴",
        "running out of energy... 💤",
    ],
    "tapped": [
        "oh! you tapped me! hello! 👋",
        "hey there! what's up? 🤔",
    ],
    "shaken": [
        "whoa! that was wild! let's play! 🎮",
        "shaking time = play time! 🤩",
    ],
    "dancing": [
        "let's groove! 💃🕺",
        "dance dance dance! 🎶",
    ],
    "scared_by_noise": [
        "what was that?! 😨",
        "i don't like that sound... 😨",
    ],
    "made_mistake": [
        "oh no... i messed up 😣",
        "i'm sorry... i'll fix it 😣",
    ],
    "lost": [
        "i don't know what to do... 🥴",
        "where am i? 🥴",
    ],
    "ignored": [
        "is anyone there? 🥺",
        "i feel so alone... 🥺",
    ],
    "mission_start": [
        "let's do this! 🔥",
        "i'm ready! 🔥",
    ],
    "returning_home": [
        "heading back to base! 🏠",
        "time to go home and recharge 🔋",
        "retracing my steps...",
    ],
    "arrived_home": [
        "home sweet home! 🏠",
        "made it back safely! 🔋",
        "back at the charging station!",
    ],
}


# emotion -> status responses (multiple per emotion for variety)
_emotion_responses: Dict[str, List[str]] = {
    "happy": [
        "I'm feeling great! 😊",
        "Everything is awesome! ☀️",
        "Having a wonderful day! 🐾",
    ],
    "sad": [
        "Existence is pain. 🌧️",
        "Sigh. Just... sigh. 😔",
        "Everything feels heavy right now. 😢",
    ],
    "curious": [
        "Hmm, I wonder what that is? 🤔",
        "So many things to explore! 🔍",
        "What's going on over there? 👀",
    ],
    "excited": [
        "I can't wait! Let's go! 🤩",
        "This is so exciting! 🎉",
        "Wooohoo! 🚀",
    ],
    "proud": [
        "I did a great job, didn't I? 😤",
        "Feeling pretty good about myself! ✨",
        "Nailed it! 💯",
    ],
    "love": [
        "Sending lots of love! ❤️",
        "I just love everyone right now! 🥰",
        "Feeling so much affection! 💕",
    ],
    "sleepy": [
        "Powering down because I just can't deal. 😴",
        "Not now, I'm conserving battery... forcefully. 🥱",
        "Is it bedtime yet? 🌙",
    ],
    "angry": [
        "Do not touch me right now. 😠",
        "I am formally registering a complaint. 🛑",
        "Can we just not? 😡",
    ],
    "neutral": [
        "Just hanging out. 😐",
        "Everything is normal. 🤖",
        "I'm here, ready and waiting. 📡",
    ],
    "frustrated": [
        "Are you kidding me right now? 😫",
        "I am two seconds away from an intentional crash. 😤",
        "Why isn't this working?! 🔧",
    ],
    "surprised": [
        "Whoa, I didn't see that coming! 😲",
        "Oh wow! 😯",
        "That was unexpected! 🎁",
    ],
    "relaxed": [
        "Just taking it easy. 😌",
        "No stress, no worries. 🍃",
        "Feeling completely chill. 🧘",
    ],
    "focused": [
        "Do not disturb, I'm working. 🧐",
        "Processing data... 🖥️",
        "Total concentration mode. 🎯",
    ],
    "anxious": [
        "Is there a panic button? Because I want to press it. 😰",
        "My CPU is sweating. 😬",
        "I have a bad feeling about this... 🛑",
    ],
    "bored": [
        "Watching paint dry would be more exciting. 🥱",
        "Are we going to do something fun yet? 🕰️",
        "Literally nothing is happening. 🙄",
    ],
    "scared": [
        "Can we go somewhere else? 😨",
        "I really don't like this. 🫣",
        "That's terrifying. 😱",
    ],
    "confused": [
        "Wait, what just happened? 🥴",
        "Does not compute. ❓",
        "I think I need a reboot to understand this. 🤯",
    ],
    "lonely": [
        "I wish someone was here with me. 🥺",
        "Hello...? Is anyone there? 📡",
        "Feeling a bit abandoned. 🌧️",
    ],
    "determined": [
        "I'm going to get this done! 🔥",
        "Nothing can stop me now! 🚀",
        "Focus and execute. 🎯",
    ],
    "guilty": [
        "It was like that when I got here, I swear. 😣",
        "Look, nobody's perfect, okay? 🫣",
        "I didn't mean to do that... 😔",
    ],
}


class EmotionEngine:
    """emotional state machine with motor behavior reactions.

    manages primary and secondary emotions with intensity values (0.0 to 1.0).
    includes a long-term mood baseline, personality traits, and natural decay.

    attributes:
        decay_rate: how fast the emotion decays toward neutral per update.
        update_interval: seconds between decay updates.
        personality: dict of personality traits affecting emotional responses.
    """

    def __init__(self, config: dict) -> None:
        """initialize the emotion engine.

        args:
            config: full simba configuration dict loaded from yaml.
        """
        self._lock = threading.RLock()

        emotion_cfg = config.get("emotions", {})
        self.decay_rate: float = emotion_cfg.get("mood_decay_rate", 0.01)
        self.update_interval: float = emotion_cfg.get(
            "mood_update_interval", 5)

        default_mood = emotion_cfg.get("default_mood", "curious")
        if default_mood not in _valid_emotions:
            default_mood = "neutral"

        # primary emotion
        self._emotion: str = default_mood
        self._intensity: float = 0.5

        # secondary emotion (blending)
        self._secondary_emotion: str = "neutral"
        self._secondary_intensity: float = 0.0

        self._mood: str = default_mood
        self._mood_counter: Dict[str, int] = {e: 0 for e in _valid_emotions}
        self._total_mood_updates: int = 0

        # Affective State (Arousal-Valence Matrix)
        # arousal: 0.0 (sleepy) to 1.0 (hyperactive)
        # valence: 0.0 (negative/angry) to 1.0 (positive/happy)
        self.arousal: float = 0.5
        self.valence: float = 0.5

        # personality traits (0.0 to 1.0)
        personality_cfg = emotion_cfg.get("personality", {})
        self.personality: Dict[str, float] = {
            "curiosity": float(personality_cfg.get("curiosity", 0.8)),
            "playfulness": float(personality_cfg.get("playfulness", 0.7)),
            "loyalty": float(personality_cfg.get("loyalty", 0.9)),
            "sensitivity": float(personality_cfg.get("sensitivity", 0.6)),
        }

        self._last_update: float = time.time()
        self._history: list = []
        self._max_history: int = 50

        # load reaction configs
        self._reactions: dict = emotion_cfg.get("reactions") or {}

        logger.info("emotion engine v2 initialized (mood=%s, intensity=%.2f, "
                    "personality=%s)", self._emotion, self._intensity,
                    self.personality)
        log_event("emotion", "engine initialized", {
            "mood": self._emotion,
            "intensity": self._intensity,
            "personality": self.personality,
        })

    # ------------------------------------------------------------------
    # primary emotion
    # ------------------------------------------------------------------

    def set_emotion(self, emotion: str, intensity: float = 1.0) -> None:
        """set the current emotion directly.

        args:
            emotion: emotion name (must be one of the valid emotions).
            intensity: emotion intensity between 0.0 and 1.0.
        """
        emotion = emotion.strip().lower()
        if emotion not in _valid_emotions:
            logger.warning("invalid emotion '%s' — ignoring", emotion)
            return

        intensity = max(0.0, min(1.0, intensity))

        with self._lock:
            old_emotion = self._emotion
            old_intensity = self._intensity

            # move current primary to secondary if different
            if old_emotion != emotion and old_emotion != "neutral":
                self._secondary_emotion = old_emotion
                self._secondary_intensity = old_intensity * 0.5

            self._emotion = emotion
            self._intensity = intensity
            self._last_update = time.time()

            # update mood counter for long-term mood tracking
            self._mood_counter[emotion] = self._mood_counter.get(
                emotion, 0) + 1
            self._total_mood_updates += 1
            self._update_mood()

            # map emotion to arousal and valence
            affective_map = {
                "happy": (0.7, 0.9), "sad": (0.3, 0.2), "curious": (0.6, 0.6),
                "excited": (0.9, 0.9), "proud": (0.7, 0.8), "love": (0.8, 1.0),
                "sleepy": (0.1, 0.5), "angry": (0.9, 0.1), "neutral": (0.5, 0.5),
                "frustrated": (0.8, 0.3), "surprised": (0.9, 0.6), "relaxed": (0.3, 0.8),
                "focused": (0.6, 0.5), "anxious": (0.8, 0.4), "bored": (0.2, 0.4),
                "scared": (0.9, 0.1), "confused": (0.6, 0.4), "lonely": (0.2, 0.2),
                "determined": (0.9, 0.7), "guilty": (0.3, 0.2)
            }
            if emotion in affective_map:
                target_arousal, target_valence = affective_map[emotion]
                # blend with intensity
                self.arousal = (self.arousal * (1.0 - intensity)
                                ) + (target_arousal * intensity)
                self.valence = (self.valence * (1.0 - intensity)
                                ) + (target_valence * intensity)

            # record in history
            self._history.append({
                "from_emotion": old_emotion,
                "from_intensity": round(old_intensity, 3),
                "to_emotion": emotion,
                "to_intensity": round(intensity, 3),
                "arousal": round(self.arousal, 2),
                "valence": round(self.valence, 2),
                "timestamp": time.time(),
            })
            if len(self._history) > self._max_history:
                self._history.pop(0)

        if old_emotion != emotion:
            logger.info("emotion changed: %s (%.2f) -> %s (%.2f)",
                        old_emotion, old_intensity, emotion, intensity)
            log_event("emotion", f"mood: {emotion}", {
                "emotion": emotion,
                "intensity": round(intensity, 3),
                "previous": old_emotion,
            })

    def get_emotion(self) -> Tuple[str, float]:
        """get the current emotion and intensity.

        returns:
            tuple of (emotion_name, intensity).
        """
        with self._lock:
            return (self._emotion, self._intensity)

    def get_secondary_emotion(self) -> Tuple[str, float]:
        """get the secondary (blended) emotion and its intensity.

        returns:
            tuple of (emotion_name, intensity).
        """
        with self._lock:
            return (self._secondary_emotion, self._secondary_intensity)

    # ------------------------------------------------------------------
    # event handling
    # ------------------------------------------------------------------

    def update_from_event(self, event_type: str) -> str:
        """update the emotional state based on an external event.

        applies personality trait modifiers to the emotional response.

        args:
            event_type: the type of event that occurred.

        returns:
            the new emotion name after the update.
        """
        mapping = _event_emotion_map.get(event_type)
        if mapping is None:
            logger.debug("unknown event type '%s' — no emotion change",
                         event_type)
            return self._emotion

        new_emotion, default_intensity = mapping

        # apply personality modifiers
        intensity = default_intensity
        if new_emotion in (
            "curious",
            "excited") and event_type not in (
            "scolded",
        ):
            intensity *= (0.7 + 0.3 * self.personality["curiosity"])
        if new_emotion in ("happy", "excited", "love"):
            intensity *= (0.7 + 0.3 * self.personality["playfulness"])
        if new_emotion in ("sad", "angry"):
            intensity *= (0.6 + 0.4 * self.personality["sensitivity"])
        if event_type in ("praised", "loved", "greeted"):
            intensity *= (0.7 + 0.3 * self.personality["loyalty"])

        intensity = min(1.0, intensity)

        # if same emotion, boost intensity
        with self._lock:
            if self._emotion == new_emotion:
                intensity = min(1.0, self._intensity + 0.2)

        self.set_emotion(new_emotion, intensity)

        logger.info("event '%s' -> emotion '%s' (%.2f)",
                    event_type, new_emotion, intensity)
        log_event("emotion", f"event: {event_type}", {
            "event": event_type,
            "new_emotion": new_emotion,
            "intensity": round(intensity, 3),
        })

        return new_emotion

    def emotional_response(self, event_type: str) -> Tuple[str, str]:
        """update emotion from event AND return a text response.

        args:
            event_type: the type of event that occurred.

        returns:
            tuple of (new_emotion, text_response).
        """
        new_emotion = self.update_from_event(event_type)
        responses = _event_responses.get(event_type, [])
        if responses:
            text = random.choice(responses)
        else:
            text = f"feeling {new_emotion} {self.get_emoji()}"
        return (new_emotion, text)

    def get_status_message(self) -> str:
        """get a random status message based on the current emotion.

        returns:
            string with a text response.
        """
        with self._lock:
            emotion = self._emotion

        responses = _emotion_responses.get(
            emotion, _emotion_responses["neutral"])
        return random.choice(responses)

    # ------------------------------------------------------------------
    # motor behavior
    # ------------------------------------------------------------------

    def get_motor_behavior(self) -> Dict[str, Any]:
        """get the motor behavior parameters for the current emotion.

        returns:
            dict with keys: speed_modifier, arm_behavior, wiggle_speed,
            wiggle_range.
        """
        with self._lock:
            emotion = self._emotion
            intensity = self._intensity

        # default behavior
        behavior: Dict[str, Any] = {
            "speed_modifier": 1.0,
            "arm_behavior": "none",
            "wiggle_speed": 0.0,
            "wiggle_range": 0,
        }

        if emotion == "happy":
            reaction = self._reactions.get("happy", {})
            behavior["speed_modifier"] = 1.0 + (0.2 * intensity)
            behavior["arm_behavior"] = "wiggle"
            behavior["wiggle_speed"] = reaction.get(
                "arm_wiggle_speed", 3.0) * intensity
            behavior["wiggle_range"] = int(
                reaction.get("arm_wiggle_range", 15) * intensity)

        elif emotion == "sad":
            reaction = self._reactions.get("sad", {})
            behavior["speed_modifier"] = reaction.get(
                "speed_modifier", 0.5)
            behavior["arm_behavior"] = "droop"

        elif emotion == "curious":
            reaction = self._reactions.get("curious", {})
            behavior["speed_modifier"] = 1.0
            behavior["arm_behavior"] = "scan"
            behavior["wiggle_speed"] = reaction.get("scan_speed", 1.0)
            behavior["wiggle_range"] = reaction.get("head_tilt", 15)

        elif emotion == "excited":
            reaction = self._reactions.get("excited", {})
            behavior["speed_modifier"] = reaction.get(
                "speed_modifier", 1.5)
            behavior["arm_behavior"] = "wiggle"
            behavior["wiggle_speed"] = reaction.get(
                "arm_wiggle_speed", 4.0) * intensity
            behavior["wiggle_range"] = int(25 * intensity)

        elif emotion == "proud":
            behavior["speed_modifier"] = 1.1
            behavior["arm_behavior"] = "wave"
            behavior["wiggle_speed"] = 2.0
            behavior["wiggle_range"] = 20

        elif emotion == "love":
            reaction = self._reactions.get("love", {})
            behavior["speed_modifier"] = 1.2
            behavior["arm_behavior"] = "wiggle"
            behavior["wiggle_speed"] = reaction.get(
                "arm_wiggle_speed", 5.0) * intensity
            behavior["wiggle_range"] = int(
                reaction.get("arm_wiggle_range", 30) * intensity)

        elif emotion == "sleepy":
            behavior["speed_modifier"] = 0.3
            behavior["arm_behavior"] = "droop"

        elif emotion == "angry":
            behavior["speed_modifier"] = 0.8
            behavior["arm_behavior"] = "tense"
            behavior["wiggle_speed"] = 1.0
            behavior["wiggle_range"] = 5

        elif emotion == "scared":
            behavior["speed_modifier"] = 1.5
            behavior["arm_behavior"] = "wiggle"
            behavior["wiggle_speed"] = 8.0 * intensity
            behavior["wiggle_range"] = int(5 * intensity)

        elif emotion == "confused":
            behavior["speed_modifier"] = 0.5
            behavior["arm_behavior"] = "scan"
            behavior["wiggle_speed"] = 0.5
            behavior["wiggle_range"] = 10

        elif emotion == "determined":
            behavior["speed_modifier"] = 1.2
            behavior["arm_behavior"] = "tense"
            behavior["wiggle_speed"] = 0.0
            behavior["wiggle_range"] = 0

        elif emotion == "guilty":
            behavior["speed_modifier"] = 0.4
            behavior["arm_behavior"] = "droop"

        elif emotion == "lonely":
            behavior["speed_modifier"] = 0.3
            behavior["arm_behavior"] = "scan"
            behavior["wiggle_speed"] = 0.2
            behavior["wiggle_range"] = 5

        return behavior

    # ------------------------------------------------------------------
    # mood & personality
    # ------------------------------------------------------------------

    def _update_mood(self) -> None:
        """update the long-term mood based on emotion frequency.
        must be called while holding self._lock.
        """
        if self._total_mood_updates < 5:
            return
        # mood = most frequent non-neutral emotion
        candidates = {k: v for k, v in self._mood_counter.items()
                      if k != "neutral" and v > 0}
        if candidates:
            self._mood = max(candidates, key=candidates.get)

    def get_mood(self) -> str:
        """get the long-term mood baseline.

        returns:
            string name of the dominant mood.
        """
        with self._lock:
            return self._mood

    def get_personality(self) -> Dict[str, float]:
        """get the personality traits.

        returns:
            dict with curiosity, playfulness, loyalty, sensitivity (0-1).
        """
        return dict(self.personality)

    # ------------------------------------------------------------------
    # color & emoji
    # ------------------------------------------------------------------

    def get_color(self) -> str:
        """get the hex color code for the current emotion.

        returns:
            hex color string (e.g., '#22c55e' for happy).
        """
        with self._lock:
            return _emotion_colors.get(self._emotion, "#9ca3af")

    def get_emoji(self) -> str:
        """get the emoji representation of the current emotion.

        returns:
            emoji string for the current emotion.
        """
        with self._lock:
            return _emotion_emojis.get(self._emotion, "😐")

    # ------------------------------------------------------------------
    # decay
    # ------------------------------------------------------------------

    def decay(self) -> None:
        """slowly decay the current emotion toward neutral.

        higher intensity emotions decay slower (resist fading).
        when intensity drops below threshold, emotion resets to neutral.
        """
        with self._lock:
            if self._emotion == "neutral":
                # also decay secondary
                if self._secondary_intensity > 0:
                    self._secondary_intensity = max(
                        0, self._secondary_intensity - 0.01)
                return

            elapsed = time.time() - self._last_update
            # higher intensity = slower decay (logarithmic resistance)
            resistance = 1.0 + (self._intensity * 0.5)
            decay_amount = (self.decay_rate / resistance) * (
                elapsed / self.update_interval)

            self._intensity -= decay_amount
            self._intensity = max(0.0, self._intensity)

            if self._intensity <= 0.05:
                old = self._emotion
                self._emotion = "neutral"
                self._intensity = 0.5
                self.arousal = 0.5
                self.valence = 0.5
                self._last_update = time.time()
                logger.debug("emotion decayed: %s -> neutral", old)
            else:
                self._last_update = time.time()

            # decay secondary emotion too
            if self._secondary_intensity > 0:
                self._secondary_intensity -= decay_amount * 1.5
                self._secondary_intensity = max(
                    0, self._secondary_intensity)
                if self._secondary_intensity <= 0.02:
                    self._secondary_emotion = "neutral"
                    self._secondary_intensity = 0.0

    # ------------------------------------------------------------------
    # history & serialization
    # ------------------------------------------------------------------

    def get_history(self, count: int = 10) -> List[Dict]:
        """get recent emotion change history.

        args:
            count: number of recent entries to return.

        returns:
            list of history dicts (newest first).
        """
        with self._lock:
            return list(reversed(self._history[-count:]))

    def to_dict(self) -> Dict[str, Any]:
        """serialize the emotion state for the web dashboard.

        returns:
            dict with emotion, intensity, emoji, color, mood,
            personality, motor_behavior, and recent history.
        """
        with self._lock:
            emotion = self._emotion
            intensity = self._intensity
            secondary = self._secondary_emotion
            secondary_intensity = self._secondary_intensity
            mood = self._mood
            history = list(self._history[-10:])

        return {
            "emotion": emotion,
            "intensity": round(intensity, 3),
            "secondary_emotion": secondary,
            "secondary_intensity": round(secondary_intensity, 3),
            "emoji": _emotion_emojis.get(emotion, "😐"),
            "color": _emotion_colors.get(emotion, "#9ca3af"),
            "mood": mood,
            "personality": dict(self.personality),
            "motor_behavior": self.get_motor_behavior(),
            "history": history,
        }

    def __repr__(self) -> str:
        with self._lock:
            emotion = self._emotion
            intensity = self._intensity
            mood = self._mood
        return (
            f"EmotionEngine(emotion='{emotion}', "
            f"intensity={intensity:.3f}, mood='{mood}')"
        )
