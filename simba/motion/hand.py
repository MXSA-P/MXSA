# _max_cyan_ — project_mxsa
"""hand controller — 3-servo finger grip in triangle formation."""

import time
import threading

try:
    import pigpio
    _HAS_PIGPIO = True
except ImportError:
    _HAS_PIGPIO = False

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.motion.hand")


class _MockPi:
    """mock pigpio for testing without hardware."""

    def set_servo_pulsewidth(self, pin, pw):
        pass

    def stop(self):
        pass


class HandController:
    """controls 3 finger servos in triangle formation for gripping.

    fingers are arranged in a triangle: one on top, two on bottom.
    camera sits between the fingers.
    """

    def __init__(self, config):
        pins = config["pins"]
        self.finger_pins = [
            pins["finger_1"],  # gpio 4
            pins["finger_2"],  # gpio 17
            pins["finger_3"],  # gpio 27
        ]

        servo_cfg = config["servos"]
        self.pulse_min = servo_cfg["pulse_min"]
        self.pulse_max = servo_cfg["pulse_max"]

        finger_cfg = servo_cfg["finger_limits"]
        self.open_angle = finger_cfg["open"]       # 30
        self.closed_angle = finger_cfg["closed"]   # 150
        self.grab_speed = finger_cfg["grab_speed"]  # 2.0

        home = servo_cfg["home_position"]
        self.finger_angles = [
            home["finger_1"],
            home["finger_2"],
            home["finger_3"],
        ]

        self._lock = threading.Lock()
        self._motion_lock = threading.Lock()
        self._grip_state = "open"

        if _HAS_PIGPIO:
            self.pi = pigpio.pi()
            if not self.pi.connected:
                logger.warning("pigpio not connected, using mock")
                self.pi = _MockPi()
        else:
            logger.warning("pigpio not available, using mock")
            self.pi = _MockPi()

        # set initial position
        for i in range(3):
            self._set_finger(i, self.finger_angles[i])

        logger.info("hand controller initialized (3-finger triangle)")

    def _angle_to_pulse(self, angle):
        angle = max(0, min(180, angle))
        return int(self.pulse_min + (angle / 180.0) *
                   (self.pulse_max - self.pulse_min))

    def _set_finger(self, finger_id, angle):
        """set individual finger to angle."""
        if 0 <= finger_id < 3:
            # Finger 3 (index 2) should not exceed 50 degrees
            if finger_id == 2:
                angle = max(0, min(50, angle))
                
            # Finger 1 (index 0) is the top finger mounted upside down relative to the bottom two
            actual_angle = (180 - angle) if finger_id == 0 else angle
            pw = self._angle_to_pulse(actual_angle)
            try:
                self.pi.set_servo_pulsewidth(self.finger_pins[finger_id], pw)
            except Exception as e:
                logger.error(f"Failed to set servo pulsewidth for finger {finger_id}: {e}")
            self.finger_angles[finger_id] = angle

    def set_finger(self, finger_id, angle):
        """set a specific finger to an angle.

        args:
            finger_id: 0, 1, or 2
            angle: servo angle (open_angle to closed_angle)
        """
        angle = max(self.open_angle, min(self.closed_angle, angle))
        with self._lock:
            self._set_finger(finger_id, angle)
        log_event("motion", f"finger {finger_id} set to {angle}°")

    def grab(self):
        """close all fingers gradually to grab an object."""
        log_event("motion", "hand grabbing!")
        with self._motion_lock:
            with self._lock:
                start_angles = list(self.finger_angles)
            distances = [self.closed_angle - start for start in start_angles]
            max_dist = max([abs(d) for d in distances] + [0])
            speed = max(0.1, self.grab_speed)
            steps = max(1, int(max_dist / speed))
    
            for step in range(1, steps + 1):
                with self._lock:
                    for f in range(3):
                        angle = start_angles[f] + (distances[f] * (step / steps))
                        self._set_finger(f, angle)
                time.sleep(0.02)
            with self._lock:
                self._grip_state = "closed"

    def adaptive_grab(self, width_percentage):
        """close fingers dynamically based on estimated object width."""
        log_event("motion", f"adaptive grab for width {width_percentage:.1f}%")
        width_percentage = max(0, min(100, width_percentage))

        # calculate target angle
        range_deg = self.closed_angle - self.open_angle
        target_angle = self.closed_angle - \
            (width_percentage / 100.0) * range_deg

        with self._motion_lock:
            with self._lock:
                start_angles = list(self.finger_angles)
                
            distances = [target_angle - start for start in start_angles]
            max_dist = max([abs(d) for d in distances] + [0])
            speed = max(0.1, self.grab_speed)
            steps = max(1, int(max_dist / speed))
    
            for step in range(1, steps + 1):
                with self._lock:
                    for f in range(3):
                        angle = start_angles[f] + (distances[f] * (step / steps))
                        self._set_finger(f, angle)
                time.sleep(0.02)
    
            with self._lock:
                self._grip_state = f"adaptive ({width_percentage:.1f}%)"

    def release(self):
        """open all fingers gradually to release object."""
        log_event("motion", "hand releasing!")
        speed = max(0.1, self.grab_speed)
        with self._motion_lock:
            while True:
                with self._lock:
                    current_angles = list(self.finger_angles)
                    all_open = not any(a > self.open_angle for a in current_angles)
                if all_open:
                    break
                    
                for i in range(3):
                    if current_angles[i] > self.open_angle:
                        current_angles[i] = max(
                            current_angles[i] - speed,
                            self.open_angle
                        )
                with self._lock:
                    for i in range(3):
                        self._set_finger(i, current_angles[i])
                time.sleep(0.02)
                
            with self._lock:
                self._grip_state = "open"
        log_event("motion", "hand grip opened")

    def point(self):
        """extend finger 1, close others — pointing gesture."""
        log_event("motion", "hand pointing!")
        with self._lock:
            self._set_finger(0, self.open_angle)      # finger 1 extended
            self._set_finger(1, self.closed_angle)     # finger 2 closed
            self._set_finger(2, self.closed_angle)     # finger 3 closed
            self._grip_state = "partial"

    def rock(self):
        """rock gesture - all fingers closed."""
        log_event("motion", "hand sign: rock")
        with self._lock:
            self._set_finger(0, self.closed_angle)
            self._set_finger(1, self.closed_angle)
            self._set_finger(2, self.closed_angle)
            self._grip_state = "rock"

    def paper(self):
        """paper gesture - all fingers open."""
        log_event("motion", "hand sign: paper")
        with self._lock:
            self._set_finger(0, self.open_angle)
            self._set_finger(1, self.open_angle)
            self._set_finger(2, self.open_angle)
            self._grip_state = "paper"

    def scissors(self):
        """scissors gesture - two fingers open, one closed."""
        log_event("motion", "hand sign: scissors")
        with self._lock:
            self._set_finger(0, self.open_angle)
            self._set_finger(1, self.open_angle)
            self._set_finger(2, self.closed_angle)
            self._grip_state = "scissors"

    def thumbs_up(self):
        """thumbs up gesture."""
        log_event("motion", "hand sign: thumbs up")
        with self._lock:
            self._set_finger(0, self.open_angle)       # thumb
            self._set_finger(1, self.closed_angle)
            self._set_finger(2, self.closed_angle)
            self._grip_state = "thumbs_up"

    def wave_fingers(self):
        """wave fingers sequentially for play mode."""
        log_event("motion", "fingers waving!")
        with self._motion_lock:
            for _ in range(3):
                for i in range(3):
                    with self._lock:
                        self._set_finger(i, self.closed_angle)
                    time.sleep(0.1)
                    with self._lock:
                        self._set_finger(i, self.open_angle)
                    time.sleep(0.1)

    def get_grip_state(self):
        """get current grip state.

        returns:
            str: 'open', 'closed', or 'partial'
        """
        with self._lock:
            return self._grip_state

    def get_positions(self):
        """get current finger positions.

        returns:
            list of 3 angles
        """
        with self._lock:
            return list(self.finger_angles)

    def cleanup(self):
        """release servos and cleanup."""
        logger.info("cleaning up hand controller")
        self.release()
        with self._lock:
            for pin in self.finger_pins:
                try:
                    self.pi.set_servo_pulsewidth(pin, 0)
                except Exception as e:
                    logger.error(f"Failed to stop servo on pin {pin}: {e}")
            if _HAS_PIGPIO and hasattr(self.pi, 'connected'):
                try:
                    self.pi.stop()
                except Exception as e:
                    logger.error(f"Failed to stop pi connection: {e}")


if __name__ == "__main__":
    import yaml
    with open("config/simba_config.yaml") as f:
        config = yaml.safe_load(f)
    hand = HandController(config)
    print("testing hand...")
    hand.grab()
    time.sleep(1)
    hand.release()
    time.sleep(1)
    hand.point()
    time.sleep(1)
    hand.wave_fingers()
    hand.cleanup()
    print("hand test complete")
