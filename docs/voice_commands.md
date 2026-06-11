# Simba Voice Commands

Below is a complete list of voice commands that Simba currently understands. For commands that take a target, you can append an object or modifier (e.g. "go forward *a little bit*", "fetch *the red ball*").

## 🦾 Physical / Motion Commands
| Action | Trigger Phrases | Description |
| :--- | :--- | :--- |
| **fetch** | "get me the...", "pass me the...", "bring me the..." | Locates and grabs a specific object, then returns. |
| **grab** | "grab that", "pick up the..." | Uses IK to immediately grab whatever is in front of the hand. |
| **hold** | "hold this", "hold it", "take this" | Closes the hand to hold an object presented to the camera. |
| **release** | "let go", "release it", "drop it" | Opens the hand to release the currently held object. |
| **forward** | "go forward", "move forward", "forward" | Drives the chassis forward. Use "a little bit" to move briefly. |
| **backward** | "go backward", "move backward", "go back" | Drives the chassis backward. Use "a little bit" to move briefly. |
| **left** | "go left", "turn left", "left" | Turns the chassis to the left. Use "a little bit" to turn briefly. |
| **right** | "go right", "turn right", "right" | Turns the chassis to the right. Use "a little bit" to turn briefly. |
| **follow** | "follow me", "come here" | Tracks a person and follows them. |
| **come** | "come to me", "over here" | Drives toward the recognized speaker. |
| **patrol** | "patrol", "explore", "guard the area" | Enters autonomous roaming/patrol mode. |
| **scan** | "scan", "look around", "survey" | Rotates to scan the environment for objects. |
| **stop** | "stop", "halt", "freeze", "stay" | Instantly halts all motion and cancels current tasks. |
| **dance** | "dance", "show me your moves" | Performs a pre-programmed sequence of dance moves. |

## 🤚 Hand Gestures
| Action | Trigger Phrases | Description |
| :--- | :--- | :--- |
| **point** | "point at that", "point" | Poses fingers to point. |
| **thumbs_up** | "thumbs up" | Poses hand in a thumbs up. |
| **rock** | "rock", "rock sign" | Poses hand in a fist/rock shape. |
| **paper** | "paper", "paper sign" | Opens hand fully. |
| **scissors** | "scissors", "scissors sign" | Poses fingers in a V-shape. |
| **wave_fingers** | "wave your fingers" | Wiggles the finger servos sequentially. |

## 🤖 Identity & Personality
| Action | Trigger Phrases | Description |
| :--- | :--- | :--- |
| **greet** | "hello", "hi simba", "good morning" | Plays a friendly greeting animation/response. |
| **who_are_you** | "who are you", "introduce yourself" | Simba explains what he is. |
| **who_am_i** | "who am i", "who is the owner" | Simba identifies his owner/creator. |
| **feeling** | "how do you feel", "are you okay" | Simba reports his current emotional state. |
| **joke** | "tell me a joke", "make me laugh" | Tells a random joke. |
| **story** | "tell me a story", "read me a story" | Recites a short story. |
| **sing** | "sing a song", "sing for me" | Sings a song. |
| **play** | "play with me", "let's play" | Enters interactive play mode. |

## 👁️ System & Vision
| Action | Trigger Phrases | Description |
| :--- | :--- | :--- |
| **status** | "what is your status", "battery level" | Reports battery, hardware connections, and system health. |
| **describe** | "what do you see", "what's around" | Uses the camera and YOLO to list detected objects in view. |
| **time** | "what time is it", "current time" | Speaks the current time. |
| **charge** | "go charge", "go to charger" | Locates charging dock and docks autonomously. |
| **rest** | "go to sleep", "shutdown" | Lowers head, goes to sleep, and powers down motors. |

## 💬 Feedback
| Action | Trigger Phrases | Description |
| :--- | :--- | :--- |
| **praise** | "good boy", "well done", "awesome" | Increases happiness and positive arousal. |
| **scold** | "bad", "no", "wrong", "stop that" | Decreases happiness and corrects behavior. |

_max_cyan_ — project_mxsa
