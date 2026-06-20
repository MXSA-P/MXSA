# _max_cyan_ — project_mxsa
"""command parser for extracting structured commands from recognized speech.

uses fuzzy keyword matching to map natural language voice input
to structured command dictionaries with action, target, and modifier fields.
all input is normalized to lowercase before processing.
"""

from difflib import SequenceMatcher
import re
from typing import Dict, List, Optional, Tuple

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.voice.command_parser")

# minimum similarity ratio for fuzzy matching
_fuzzy_threshold = 0.70


def _similarity(a: str, b: str) -> float:
    """compute similarity ratio between two strings.

    args:
        a: first string.
        b: second string.

    returns:
        similarity ratio between 0.0 and 1.0.
    """
    return SequenceMatcher(None, a, b).ratio()


# command pattern definitions: list of (patterns, action, has_target)
# patterns are checked in order; first match wins
_command_patterns: List[Tuple[List[str], str, bool]] = [
    # clear path commands
    (
        ["clear path", "reset path", "forget the way home", "clear breadcrumbs"],
        "clear_path",
        False,
    ),
    # path status commands
    (
        ["how far from home", "path status", "how many steps", "where am i"],
        "path_status",
        False,
    ),
    # speed up commands
    (
        ["speed up", "go faster", "faster"],
        "speed_up",
        False,
    ),
    # slow down commands
    (
        ["slow down", "go slower", "slower"],
        "slow_down",
        False,
    ),
    # calibrate commands
    (
        ["calibrate", "calibrate servos", "reset servos", "calibrate arm"],
        "calibrate",
        False,
    ),
    # temperature commands
    (
        ["temperature", "how hot are you", "cpu temperature",
         "system temperature"],
        "temperature",
        False,
    ),
    # fetch commands — extract target object
    (
        ["get me the", "pass me the", "bring me the",
         "fetch the", "hand me the", "give me the"],
        "fetch",
        True,
    ),
    # grab commands (IK physical grasp)
    (
        ["grab that", "grab the", "pick up the", "grab it", "pick it up"],
        "grab",
        True,
    ),
    # identity & personality
    (
        ["who am i",
         "what is my name",
         "who is the owner",
         "who is my owner",
         "who is your owner",
         "who created you"],
        "who_am_i",
        False,
    ),
    (
        ["who are you", "what are you", "introduce yourself"],
        "who_are_you",
        False,
    ),
    (
        ["tell me a joke", "make me laugh", "joke"],
        "joke",
        False,
    ),
    (
        ["what time is it", "tell me the time", "current time"],
        "time",
        False,
    ),
    (
        ["sing a song", "sing for me", "sing something", "sing"],
        "sing",
        False,
    ),
    (
        ["tell me a story", "read me a story", "story time"],
        "story",
        False,
    ),
    (
        ["what is your status",
         "how are you",
         "report status",
         "battery level",
         "status report"],
        "status",
        False,
    ),
    (
        ["patrol", "explore", "look around", "go on patrol", "start patrolling"],
        "patrol",
        False,
    ),
    (
        ["dance", "do a dance", "show me your moves"],
        "dance",
        False,
    ),
    (
        ["follow me", "come here", "follow"],
        "follow",
        False,
    ),
    # charge commands
    (
        ["go charge", "charge yourself", "go to charger", "charge up",
         "recharge", "go recharge"],
        "charge",
        False,
    ),
    # scan commands
    (
        ["scan", "look around", "scan the area", "scan around",
         "survey", "check around", "what's around you"],
        "scan",
        False,
    ),
    # play commands
    (
        ["play with me", "let's play", "lets play", "wanna play",
         "play time", "playtime", "time to play"],
        "play",
        False,
    ),
    # greet commands
    (
        ["hello", "hi", "hey simba", "hi simba", "hello simba",
         "hey there", "good morning", "good evening", "howdy"],
        "greet",
        False,
    ),
    # love commands
    (
        ["i love you", "love you simba", "love you", "you're the best",
         "you are the best", "i adore you", "you're awesome"],
        "love",
        False,
    ),
    # stop commands
    (
        ["stop", "halt", "freeze", "don't move", "stay", "wait",
         "hold on", "pause"],
        "stop",
        False,
    ),
    # hold commands
    (
        ["hold this", "hold it", "take this", "grab this", "hold"],
        "hold",
        True,
    ),
    # release commands
    (
        ["that's fine", "let go", "release it",
            "drop it", "thats fine", "release"],
        "release",
        True,
    ),
    # directional commands
    (
        ["go forward", "move forward", "forward"],
        "forward",
        True,
    ),
    (
        ["go backward", "move backward", "backward", "go back", "move back"],
        "backward",
        True,
    ),
    (
        ["go left", "move left", "turn left", "left"],
        "left",
        True,
    ),
    (
        ["go right", "move right", "turn right", "right"],
        "right",
        True,
    ),
    # come commands
    (
        ["come here", "come to me", "come over", "come closer",
         "over here", "this way"],
        "come",
        False,
    ),
    # describe commands
    (
        ["what do you see", "what's around", "whats around",
         "describe", "what can you see", "tell me what you see",
         "what is around you", "what are you looking at"],
        "describe",
        False,
    ),
    # praise commands
    (
        ["good boy", "well done", "nice", "great job", "awesome",
         "perfect", "good job", "excellent", "bravo", "nice work",
         "amazing", "that's great", "wonderful"],
        "praise",
        False,
    ),
    # scold commands
    (
        ["bad", "no", "wrong", "bad boy", "stop that", "don't do that",
         "not like that", "incorrect", "that's wrong", "nope"],
        "scold",
        False,
    ),
    # dance commands
    (
        ["dance", "do a dance", "show me some moves", "let's dance",
         "dance for me", "break it down"],
        "dance",
        False,
    ),
    # status commands
    (
        ["what is your status", "how are you", "how are you doing",
         "battery level", "are you ok", "report status"],
        "status",
        False,
    ),
    # patrol commands
    (
        ["patrol", "go on patrol", "guard the area", "start patrolling",
         "keep watch"],
        "patrol",
        False,
    ),
    # rest commands
    (
        ["go to sleep", "rest now", "shutdown", "power off", "go rest"],
        "rest",
        False,
    ),
    # feeling commands
    (
        ["how do you feel", "are you okay", "are you ok",
            "how are you feeling", "what are you feeling"],
        "feeling",
        False,
    ),
    # hand gesture commands
    (
        ["point at that", "point there", "point"],
        "point",
        False,
    ),
    (
        ["thumbs up", "give me a thumbs up"],
        "thumbs_up",
        False,
    ),
    (
        ["rock", "make a rock", "rock sign"],
        "rock",
        False,
    ),
    (
        ["paper", "make paper", "paper sign"],
        "paper",
        False,
    ),
    (
        ["scissors", "make scissors", "scissors sign"],
        "scissors",
        False,
    ),
    (
        ["wave your fingers", "finger wave", "wiggle your fingers"],
        "wave_fingers",
        False,
    ),
]


# precompile regexes to avoid runtime compilation
_PUNCTUATION_RE = re.compile(r"[^\w\s\'-]")
_WHITESPACE_RE = re.compile(r"\s+")

_fillers_list = [
    "um", "uh", "like", "you know", "please", "could you",
    "can you", "would you", "simba"
]
_fillers_list.sort(key=len, reverse=True)
_fillers_regex_str = "|".join(re.escape(f) for f in _fillers_list)
_FILLER_RE = re.compile(rf"(?<![\w'-])(?:{_fillers_regex_str})(?![\w'-])", flags=re.IGNORECASE)


_compiled_patterns = {}
for _patterns, _, _ in _command_patterns:
    for _pattern in _patterns:
        _compiled_patterns[_pattern] = re.compile(rf"(?<![\w'-]){re.escape(_pattern)}(?![\w'-])")


class CommandParser:
    """extracts structured commands from recognized speech text.

    uses a combination of exact substring matching and fuzzy matching
    to identify commands from natural language input. all input is
    normalized to lowercase.

    usage:
        parser = CommandParser()
        result = parser.parse("hey simba get me the bottle")
        # result = {"action": "fetch", "target": "bottle", "modifier": None}
    """

    def __init__(self) -> None:
        """initialize the command parser."""
        self._parse_count: int = 0
        self._match_count: int = 0
        logger.info("command parser initialized with %d command patterns",
                    len(_command_patterns))

    def parse(self, text: str) -> Optional[Dict[str, Optional[str]]]:
        """parse recognized text into a structured command.

        args:
            text: recognized speech text (will be normalized to lowercase).

        returns:
            dict with keys 'action', 'target', 'modifier' if a command
            was matched, or none if no command was recognized.
        """
        if not text or not text.strip():
            return None

        # prevent algorithmic complexity dos on massive strings
        if len(text) > 2000:
            logger.warning("input text too long (%d chars), discarding", len(text))
            return None

        self._parse_count += 1
        normalized = text.strip().lower()

        # remove common filler words for better matching
        cleaned = self._clean_text(normalized)

        # try exact substring matching first (faster)
        result = self._try_exact_match(cleaned, normalized)
        if result is not None:
            self._match_count += 1
            logger.info("command parsed: %s (exact match from '%s')",
                        result, text)
            log_event("voice", "command parsed", {
                "text": text, "command": result, "method": "exact"
            })
            return result

        # fall back to fuzzy matching
        result = self._try_fuzzy_match(cleaned, normalized)
        if result is not None:
            self._match_count += 1
            logger.info("command parsed: %s (fuzzy match from '%s')",
                        result, text)
            log_event("voice", "command parsed", {
                "text": text, "command": result, "method": "fuzzy"
            })
            return result

        logger.debug("no command matched for: '%s'", text)
        return None

    def _clean_text(self, text: str) -> str:
        """remove common filler words and extra whitespace."""
        # remove punctuation and collapse spaces
        result = _PUNCTUATION_RE.sub('', text)
        result = _WHITESPACE_RE.sub(' ', result).strip()
        
        # remove fillers using precompiled regex
        result = _FILLER_RE.sub(" ", result)
        
        # collapse multiple spaces again
        result = _WHITESPACE_RE.sub(' ', result).strip()
        return result

    def _try_exact_match(
        self, cleaned: str, original: str
    ) -> Optional[Dict[str, Optional[str]]]:
        """attempt exact substring matching against command patterns.

        args:
            cleaned: cleaned text with fillers removed.
            original: original normalized text.

        returns:
            command dict if matched, none otherwise.
        """
        for patterns, action, has_target in _command_patterns:
            for pattern in patterns:
                pattern_regex = _compiled_patterns[pattern]
                if pattern_regex.search(cleaned) or pattern_regex.search(original):
                    target = None
                    modifier = None

                    if has_target:
                        target = self._extract_target(cleaned, pattern)
                        if target is None:
                            target = self._extract_target(original, pattern)

                    return {
                        "action": action,
                        "target": target,
                        "modifier": modifier,
                    }
        return None

    def _try_fuzzy_match(
        self, cleaned: str, original: str
    ) -> Optional[Dict[str, Optional[str]]]:
        """attempt fuzzy matching against command patterns.

        compares input text segments against known patterns using
        sequence matcher similarity ratio.

        args:
            cleaned: cleaned text with fillers removed.
            original: original normalized text.

        returns:
            command dict if a sufficiently similar match is found, none otherwise.
        """
        best_score: float = 0.0
        best_action: Optional[str] = None
        best_has_target: bool = False
        best_pattern: str = ""

        words = cleaned.split()

        for patterns, action, has_target in _command_patterns:
            for pattern in patterns:
                pattern_words = pattern.split()
                pattern_len = len(pattern_words)

                # slide a window of pattern length over the input words
                for i in range(max(1, len(words) - pattern_len + 1)):
                    window = " ".join(words[i:i + pattern_len])
                    score = _similarity(window, pattern)

                    if score > best_score and score >= _fuzzy_threshold:
                        best_score = score
                        best_action = action
                        best_has_target = has_target
                        best_pattern = window

                # also compare against the full cleaned text
                score = _similarity(cleaned, pattern)
                if score > best_score and score >= _fuzzy_threshold:
                    best_score = score
                    best_action = action
                    best_has_target = has_target
                    best_pattern = pattern

        if best_action is not None:
            target = None
            if best_has_target:
                target = self._extract_target(cleaned, best_pattern)
                if target is None:
                    target = self._extract_target(original, best_pattern)

            return {
                "action": best_action,
                "target": target,
                "modifier": None,
            }

        return None

    def _extract_target(self, text: str, pattern: str) -> Optional[str]:
        """extract target object from text after matching pattern.

        args:
            text: full text containing the pattern and target.
            pattern: matched command pattern.

        returns:
            target string if found, none otherwise.
        """
        pattern_regex = _compiled_patterns.get(pattern)
        if pattern_regex:
            match = pattern_regex.search(text)
        else:
            match = re.search(rf"(?<![\w'-]){re.escape(pattern)}(?![\w'-])", text)

        if not match:
            return None
        
        remainder = text[match.end():].strip()

        # remove trailing filler words efficiently using string manipulation
        # to prevent O(N^2) regex behavior on massive strings
        trailing_fillers = ("please", "now", "quickly", "fast", "right now", "for me")
        end_idx = len(remainder)
        remainder_lower = remainder.lower()
        
        while end_idx > 0:
            matched = False
            for filler in trailing_fillers:
                filler_len = len(filler)
                if end_idx >= filler_len and remainder_lower[end_idx - filler_len : end_idx] == filler:
                    idx = end_idx - filler_len
                    if idx == 0 or not (remainder_lower[idx - 1].isalnum() or remainder_lower[idx - 1] in ("'", "-", "_")):
                        end_idx = idx
                        matched = True
                        while end_idx > 0 and remainder_lower[end_idx - 1].isspace():
                            end_idx -= 1
                        break
            if not matched:
                break

        if end_idx > 0:
            return remainder[:end_idx]
        return None

    def get_stats(self) -> Dict[str, int]:
        """return parsing statistics.

        returns:
            dict with parse_count and match_count.
        """
        return {
            "parse_count": self._parse_count,
            "match_count": self._match_count,
        }

    def __repr__(self) -> str:
        """return string representation of the parser."""
        return (
            f"CommandParser(parsed={self._parse_count}, "
            f"matched={self._match_count})"
        )
