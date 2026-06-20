# _max_cyan_ — project_mxsa
"""Arm controller — 4-servo articulated arm using pigpio."""

import functools
import math
import threading
import time
from typing import Any, Dict, Tuple  # Cleaned typing

try:
    import pigpio
    _HAS_PIGPIO = True
except ImportError:
    _HAS_PIGPIO = False

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.motion.arm")


class _MockPi:
    """Mock pigpio interface for testing without hardware."""

    def set_servo_pulsewidth(self, pin, pw):
        """Set the pulsewidth for the specified servo pin."""
        pass

    def stop(self):
        """Stop the mock pigpio interface."""
        pass


@functools.lru_cache(maxsize=1024)
def _calculate_ik(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Calculate inverse kinematics with caching to reduce CPU load."""
    if x is None or y is None or z is None:
        raise ValueError("XYZ coordinates cannot be None")

    try:
        x = float(x)
        y = float(y)
        z = float(z)
    except (ValueError, TypeError) as e:
        raise ValueError(f"XYZ coordinates must be numeric: {e}")

    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        raise ValueError("XYZ coordinates must be finite numbers")

    MAX_COORD = 100.0
    if not (-MAX_COORD <= x <= MAX_COORD and
            -MAX_COORD <= y <= MAX_COORD and
            -MAX_COORD <= z <= MAX_COORD):
        x = max(-MAX_COORD, min(MAX_COORD, x))
        y = max(-MAX_COORD, min(MAX_COORD, y))
        z = max(-MAX_COORD, min(MAX_COORD, z))

    L1 = 15.0  # length from base to wrist (cm)
    L2 = 10.0  # length from wrist to fingertip (cm)

    # 1. Base Rotation (Y-axis twist)
    rotation_rad = math.atan2(x, y)
    rot_angle = 90 - math.degrees(rotation_rad)

    # 2. Planar IK (r, z)
    r = math.sqrt(x**2 + y**2)

    # safe distance check
    target_dist = math.sqrt(r**2 + z**2)
    if target_dist > (L1 + L2):
        logger.warning("Target xyz(%.1f, %.1f, %.1f) is unreachable. Clamping.", x, y, z)
        scale = (L1 + L2 - 0.1) / target_dist
        r *= scale
        z *= scale
    elif target_dist < abs(L1 - L2):
        logger.warning("Target xyz(%.1f, %.1f, %.1f) is too close. Clamping.", x, y, z)
        if target_dist < 0.001:
            r = abs(L1 - L2) + 0.1
            z = 0.0
        else:
            scale = (abs(L1 - L2) + 0.1) / target_dist
            r *= scale
            z *= scale

    # calculate wrist angle (theta2) using cosine rule
    c2 = (r**2 + z**2 - L1**2 - L2**2) / (2 * L1 * L2)

    # strictly clamp c2 to [-1, 1] to prevent math domain errors
    if math.isnan(c2):
        c2 = 1.0
    elif c2 > 1.0:
        c2 = 1.0
    elif c2 < -1.0:
        c2 = -1.0

    theta2_rad = math.acos(c2)

    # calculate elbow angle (theta1)
    k1 = L1 + L2 * c2
    k2 = L2 * math.sin(theta2_rad)
    theta1_rad = math.atan2(z, r) - math.atan2(k2, k1)

    # convert to degrees and map to servo ranges
    elbow_angle = 180 - math.degrees(theta1_rad)
    elbow_2_offset = math.degrees(theta2_rad)

    return rot_angle, elbow_angle, elbow_2_offset


class ArmController:
    """Controls 4-servo arm: rotation (y-axis), elbow (up/down), elbow_2, wrist (up/down).

    Uses pigpio for hardware-timed pwm to avoid servo jitter.
    All movements are smooth-interpolated for natural motion.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize arm controller.

        Args:
            config: dict from simba_config.yaml
        """
        pins = config["pins"]
        self.rotation_pin = pins["arm_rotation"]  # gpio 22
        self.elbow_pin = pins["arm_elbow"]         # gpio 23
        self.elbow_2_pin = pins["arm_elbow_2"]     # gpio 25
        self.wrist_pin = pins["arm_wrist"]         # gpio 24

        servo_cfg = config["servos"]
        self.pulse_min = servo_cfg["pulse_min"]     # 500
        self.pulse_max = servo_cfg["pulse_max"]     # 2500
        self.servo_max_angle = servo_cfg.get("max_angle", 180.0)

        limits = servo_cfg["arm_limits"]
        self.rotation_min = limits["rotation_min"]  # 0
        self.rotation_max = limits["rotation_max"]  # 180
        self.elbow_min = limits["elbow_min"]        # 10
        self.elbow_max = limits["elbow_max"]        # 170
        self.elbow_2_min = limits.get("elbow_2_min", 10)
        self.elbow_2_max = limits.get("elbow_2_max", 170)
        self.wrist_min = limits["wrist_min"]        # 0
        self.wrist_max = limits["wrist_max"]        # 150
        self.move_speed = limits["move_speed"]      # 1.5 deg/step

        home = servo_cfg["home_position"]
        self.home_angles = {
            "rotation": home["arm_rotation"],  # 90
            "elbow": home["arm_elbow"],        # 90
            "elbow_2": home.get("arm_elbow_2", 90),
            "wrist": home["arm_wrist"],        # 90
        }

        # current positions
        self.current = {
            "rotation": self.home_angles["rotation"],
            "elbow": self.home_angles["elbow"],
            "elbow_2": self.home_angles["elbow_2"],
            "wrist": self.home_angles["wrist"],
        }

        self._lock = threading.Lock()
        self._motion_lock = threading.Lock()
        self._moving = False
        self._stop_event = threading.Event()

        # initialize pigpio
        if _HAS_PIGPIO:
            self.pi = pigpio.pi()
            if not self.pi.connected:
                logger.warning("pigpio daemon not running, using mock")
                self.pi = _MockPi()
        else:
            logger.warning("pigpio not available, using mock")
            self.pi = _MockPi()

        # configure servo pins as outputs
        if _HAS_PIGPIO:
            try:
                for pin in [self.rotation_pin, self.elbow_pin, self.elbow_2_pin, self.wrist_pin]:
                    self.pi.set_mode(pin, pigpio.OUTPUT)
            except Exception as e:
                logger.error("pigpio mode setting error: %s", e)

        # move to home position
        self.home()
        logger.info("arm controller initialized")

    def _angle_to_pulse(self, angle: float) -> int:
        """Convert angle to pulse width (500-2500 microseconds)."""
        angle = max(0, min(self.servo_max_angle, angle))
        return int(round(self.pulse_min + (angle / self.servo_max_angle) *
                   (self.pulse_max - self.pulse_min)))

    def _set_servo(self, pin: int, angle: float) -> None:
        """Set a servo to a specific angle."""
        pw = self._angle_to_pulse(angle)
        try:
            self.pi.set_servo_pulsewidth(pin, pw)
        except Exception as e:
            logger.error("pigpio error setting pin %d to pw %d: %s", pin, pw, e)

    def _move_smooth(self, pin, current_angle, target_angle, speed=None):
        """Smoothly interpolate servo from current to target angle with ease-in-out."""
        if speed is None:
            speed = self.move_speed

        speed = max(0.1, speed)

        distance = target_angle - current_angle
        if abs(distance) < 0.1:
            return target_angle

        steps = max(1, int(abs(distance) / speed))

        for i in range(1, steps + 1):
            t = i / steps
            ease_t = 6 * (t ** 5) - 15 * (t ** 4) + 10 * (t ** 3)
            angle = current_angle + distance * ease_t
            with self._lock:
                self._set_servo(pin, angle)
            if self._stop_event.wait(0.02):
                return angle

        # final position
        with self._lock:
            self._set_servo(pin, target_angle)
        return target_angle

    def rotation(self, angle):
        """Rotate arm left/right."""
        angle = max(self.rotation_min, min(self.rotation_max, angle))
        with self._motion_lock:
            with self._lock:
                self._moving = True
                current_angle = self.current["rotation"]

            final_angle = self._move_smooth(
                self.rotation_pin, current_angle, angle
            )

            with self._lock:
                self.current["rotation"] = final_angle
                self._moving = False

        log_event("motion", f"arm rotated to {angle}°")

    def raise_arm(self, angle):
        """Move arm elbows up/down simultaneously.

        Args:
            angle: target angle (elbow_min to elbow_max)
        """
        angle1 = max(self.elbow_min, min(self.elbow_max, angle))
        # Invert elbow 2 since it's physically mirrored
        inverted_angle = 180 - angle1
        angle2 = max(self.elbow_2_min, min(self.elbow_2_max, inverted_angle))
        self.move_smooth({"elbow": angle1, "elbow_2": angle2})
        log_event("motion", f"arm elbows to {angle}° (elbow_2 inverted to {angle2}°)")

    def wrist(self, angle):
        """Move wrist up/down."""
        angle = max(self.wrist_min, min(self.wrist_max, angle))
        with self._motion_lock:
            with self._lock:
                self._moving = True
                current_angle = self.current["wrist"]

            final_angle = self._move_smooth(
                self.wrist_pin, current_angle, angle
            )

            with self._lock:
                self.current["wrist"] = final_angle
                self._moving = False

        log_event("motion", f"wrist to {angle}°")

    def home(self):
        """Return all servos to home position."""
        self.move_smooth({
            "rotation": self.home_angles["rotation"],
            "elbow": self.home_angles["elbow"],
            "elbow_2": self.home_angles["elbow_2"],
            "wrist": self.home_angles["wrist"]
        })
        log_event("motion", "arm returned to home position")

    def move_smooth(self, target_angles, speed=None):
        """Move all arm servos simultaneously to target angles."""
        if speed is None:
            speed = self.move_speed
        speed = max(0.1, speed)

        with self._motion_lock:
            with self._lock:
                self._moving = True
                start_angles = {k: self.current[k] for k in self.current}

        targets = {
            "rotation": max(
                self.rotation_min, min(
                    self.rotation_max, target_angles.get(
                        "rotation", start_angles["rotation"]))), "elbow": max(
                self.elbow_min, min(
                    self.elbow_max, target_angles.get(
                        "elbow", start_angles["elbow"]))), "elbow_2": max(
                self.elbow_2_min, min(
                    self.elbow_2_max, target_angles.get(
                        "elbow_2", start_angles["elbow_2"]))), "wrist": max(
                self.wrist_min, min(
                    self.wrist_max, target_angles.get(
                        "wrist", start_angles["wrist"]))), }

        steps = max(
            abs(targets["rotation"] - start_angles["rotation"]),
            abs(targets["elbow"] - start_angles["elbow"]),
            abs(targets["elbow_2"] - start_angles["elbow_2"]),
            abs(targets["wrist"] - start_angles["wrist"]),
        )
        num_steps = max(1, int(steps / speed))

        interrupted = False
        for i in range(1, num_steps + 1):
            t = i / num_steps
            ease_t = 6 * (t ** 5) - 15 * (t ** 4) + 10 * (t ** 3)
            with self._lock:
                for key, pin in [("rotation", self.rotation_pin),
                                 ("elbow", self.elbow_pin),
                                 ("elbow_2", self.elbow_2_pin),
                                 ("wrist", self.wrist_pin)]:
                    angle = start_angles[key] + ease_t * \
                        (targets[key] - start_angles[key])
                    self._set_servo(pin, angle)
                    self.current[key] = angle

            if self._stop_event.wait(0.02):
                interrupted = True
                with self._lock:
                    for key in targets:
                        self.current[key] = start_angles[key] + \
                            ease_t * (targets[key] - start_angles[key])
                    self._moving = False
                return

        # set final positions
        with self._lock:
            if not interrupted:
                for key in targets:
                    self.current[key] = targets[key]
            self._moving = False

    def move_to_xyz(self, x: float, y: float, z: float, wrist_roll: float = None) -> None:
        """Move arm to xyz coordinate using inverse kinematics.

        wrist_roll: optional angle (0-180), 180=fingers up, 0=fingers down.
                    if None, maintains current wrist orientation.

        Args:
            x, y, z: float coordinates in cm
        """
        if x is None or y is None or z is None:
            logger.error("move_to_xyz: Target coordinates cannot be None")
            return

        try:
            x = float(x)
            y = float(y)
            z = float(z)
        except (ValueError, TypeError) as e:
            logger.error("move_to_xyz: Target coordinates must be numeric. %s", e)
            return

        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            logger.error("move_to_xyz: Target coordinates must be finite numbers.")
            return

        MAX_COORD = 100.0
        if not (-MAX_COORD <= x <= MAX_COORD and
                -MAX_COORD <= y <= MAX_COORD and
                -MAX_COORD <= z <= MAX_COORD):
            logger.warning(
                "move_to_xyz: Target coordinates xyz(%.1f, %.1f, %.1f) "
                "wildly out of bounds. Clamping.", x, y, z)
            x = max(-MAX_COORD, min(MAX_COORD, x))
            y = max(-MAX_COORD, min(MAX_COORD, y))
            z = max(-MAX_COORD, min(MAX_COORD, z))

        # Round coordinates to 1 decimal place (1 mm precision) to maximize cache hits
        x = round(x, 1)
        y = round(y, 1)
        z = round(z, 1)

        rot_angle, elbow_angle, elbow_2_offset = _calculate_ik(x, y, z)

        # elbow_2 acts as the second pitch joint
        elbow_2_angle = self.home_angles.get("elbow_2", 90) + elbow_2_offset

        if wrist_roll is not None:
            try:
                wrist_angle = float(wrist_roll)
                if not math.isfinite(wrist_angle):
                    wrist_angle = self.current.get("wrist", 90)
            except (ValueError, TypeError):
                logger.warning(
                    "move_to_xyz: wrist_roll could not be cast to float. Using current angle.")
                wrist_angle = self.current.get("wrist", 90)
        else:
            wrist_angle = self.current.get("wrist", 90)

        log_event(
            "motion", f"IK calculated: rot={
                rot_angle:.1f}, elbow={
                elbow_angle:.1f}, elbow_2={
                elbow_2_angle:.1f}, wrist={
                wrist_angle:.1f} for xyz({x},{y},{z})")

        # move arm smoothly to calculated angles
        self.move_smooth({
            "rotation": rot_angle,
            "elbow": elbow_angle,
            "elbow_2": elbow_2_angle,
            "wrist": wrist_angle
        })

    def wave(self):
        """Wave the arm for greeting."""
        log_event("motion", "simba is waving!")
        self.raise_arm(150)
        for _ in range(3):
            if self._stop_event.is_set():
                break
            self.wrist(30)
            if self._stop_event.wait(0.2):
                break
            self.wrist(130)
            if self._stop_event.wait(0.2):
                break
        if not self._stop_event.is_set():
            self.home()

    def wiggle(self, speed=3.0, angle_range=20, duration=2.0):
        """Wiggle the arm excitedly (for emotions like love/excitement).

        Args:
            speed: wiggle speed (degrees per step)
            angle_range: wiggle amplitude in degrees
            duration: how long to wiggle in seconds
        """
        log_event(
            "motion",
            f"arm wiggling! speed={speed}, range={angle_range}")
        start_time = time.time()
        center = self.current["rotation"]

        with self._motion_lock:
            with self._lock:
                self._moving = True

            while time.time() - start_time < duration:
                target_high = max(
                    self.rotation_min, min(
                        self.rotation_max, center + angle_range))
                target_low = max(
                    self.rotation_min, min(
                        self.rotation_max, center - angle_range))

                if self._stop_event.is_set():
                    break
                final = self._move_smooth(
                    self.rotation_pin, self.current["rotation"], target_high, speed)
                with self._lock:
                    self.current["rotation"] = final

                if self._stop_event.is_set():
                    break
                final = self._move_smooth(
                    self.rotation_pin, self.current["rotation"], target_low, speed)
                with self._lock:
                    self.current["rotation"] = final

            # return to center
            if not self._stop_event.is_set():
                final = self._move_smooth(
                    self.rotation_pin, self.current["rotation"], center, speed)
                with self._lock:
                    self.current["rotation"] = final

            with self._lock:
                self._moving = False

    def handshake(self):
        """Extend arm forward and do a handshake motion."""
        log_event("motion", "simba wants to shake hands!")
        self.raise_arm(120)
        self.wrist(90)
        if self._stop_event.wait(0.3):
            return
        # handshake up-down motion
        for _ in range(3):
            if self._stop_event.is_set():
                break
            self.raise_arm(130)
            if self._stop_event.wait(0.15):
                break
            self.raise_arm(110)
            if self._stop_event.wait(0.15):
                break
        if not self._stop_event.is_set():
            self.home()

    def droop(self, angle=30):
        """Droop the arm down (for sadness)."""
        log_event("motion", "arm drooping (sad)")
        self.raise_arm(self.elbow_min + angle)
        self.wrist(self.wrist_min + 20)

    def nod(self):
        """Nod the arm up and down (for agreement)."""
        log_event("motion", "simba is nodding")
        center = self.current["elbow"]
        for _ in range(2):
            self.raise_arm(center + 20)
            self.raise_arm(center - 20)
        self.raise_arm(center)

    def shake_head(self):
        """Shake the arm side to side (for disagreement)."""
        log_event("motion", "simba is shaking head")
        center = self.current["rotation"]
        for _ in range(2):
            self.rotation(center + 30)
            self.rotation(center - 30)
        self.rotation(center)

    def celebrate(self):
        """Celebrate excitedly!"""
        log_event("motion", "simba is celebrating!")
        self.raise_arm(self.elbow_max - 20)
        self.wrist(self.wrist_max - 20)
        for _ in range(3):
            if self._stop_event.is_set():
                break
            self.rotation(self.home_angles["rotation"] + 45)
            self.rotation(self.home_angles["rotation"] - 45)
        if not self._stop_event.is_set():
            self.home()

    def get_position(self):
        """Get current arm position.

        Returns:
            dict with current angles
        """
        with self._lock:
            return dict(self.current)

    def is_moving(self):
        """Check if arm is currently moving."""
        with self._lock:
            return self._moving

    def cleanup(self):
        """Release all servos and cleanup pigpio."""
        logger.info("cleaning up arm controller")
        self._stop_event.set()

        with self._motion_lock:
            with self._lock:
                for pin in [
                        self.rotation_pin,
                        self.elbow_pin,
                        self.elbow_2_pin,
                        self.wrist_pin]:
                    self.pi.set_servo_pulsewidth(pin, 0)
                if _HAS_PIGPIO and hasattr(self.pi, 'connected'):
                    self.pi.stop()


if __name__ == "__main__":
    # test mode
    import yaml
    with open("config/simba_config.yaml") as f:
        config = yaml.safe_load(f)
    arm = ArmController(config)
    print("testing arm movements...")
    arm.wave()
    time.sleep(1)
    arm.wiggle(speed=3, angle_range=15, duration=2)
    time.sleep(1)
    arm.handshake()
    arm.cleanup()
    print("arm test complete")
