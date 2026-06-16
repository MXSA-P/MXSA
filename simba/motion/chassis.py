# _max_cyan_ — project_mxsa
"""chassis controller — 2wd differential drive via L298N motor driver."""

import time
import threading

try:
    import pigpio
    _HAS_PIGPIO = True
except ImportError:
    _HAS_PIGPIO = False

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.motion.chassis")


class _MockPi:
    def write(self, pin, val): pass
    def set_PWM_dutycycle(self, pin, dc): pass
    def set_PWM_frequency(self, pin, freq): pass
    def set_PWM_range(self, pin, range_val): pass
    def stop(self): pass
    def brake(self): pass


from typing import Dict, Any, Optional

class ChassisController:
    """2wd differential drive with 6-pin L298N motor driver module.

    motor a = left wheel, motor b = right wheel.
    ball caster provides front support.
    Uses ENA/ENB for PWM speed control, and IN1-IN4 for digital direction logic.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        pins = config["pins"]
        self.in1 = pins["motor_a_in1"]       # gpio 5
        self.in2 = pins["motor_a_in2"]       # gpio 6
        self.ena = pins.get("motor_a_en", 12)
        self.in3 = pins["motor_b_in3"]       # gpio 16
        self.in4 = pins["motor_b_in4"]       # gpio 26
        self.enb = pins.get("motor_b_en", 13)

        motor_cfg = config["motors"]
        self.max_speed = motor_cfg["max_speed"]
        self.cruise_speed = motor_cfg["cruise_speed"]
        self.turn_speed = motor_cfg["turn_speed"]
        self.slow_speed = motor_cfg["slow_speed"]

        self._lock = threading.Lock()
        self._motion_lock = threading.Lock()
        self._current_speed = (0, 0)
        self._stop_event = threading.Event()

        if _HAS_PIGPIO:
            self.pi = pigpio.pi()
            if not self.pi.connected:
                self.pi = _MockPi()
        else:
            self.pi = _MockPi()

        # set pin modes and frequency
        try:
            for pin in [self.in1, self.in2, self.in3, self.in4]:
                self.pi.write(pin, 0)

            for pin in [self.ena, self.enb]:
                self.pi.set_PWM_frequency(pin, 1000)
                self.pi.set_PWM_range(pin, 100)
                self.pi.set_PWM_dutycycle(pin, 0)
        except Exception as e:
            logger.error("pigpio hardware init error: %s", e)

        logger.info(
            "chassis controller initialized (6-pin L298N 2wd differential)")

    def _set_motor_a(self, speed: float) -> None:
        """set left motor speed. positive = forward, negative = backward."""
        duty = int(round(abs(speed)))  # 0-100
        try:
            self.pi.set_PWM_dutycycle(self.ena, min(duty, 100))
            if speed > 0:
                self.pi.write(self.in1, 1)
                self.pi.write(self.in2, 0)
            elif speed < 0:
                self.pi.write(self.in1, 0)
                self.pi.write(self.in2, 1)
            else:
                self.pi.write(self.in1, 0)
                self.pi.write(self.in2, 0)
        except Exception as e:
            logger.error("pigpio motor_a error: %s", e)

    def _set_motor_b(self, speed: float) -> None:
        """set right motor speed. positive = forward, negative = backward."""
        duty = int(round(abs(speed)))
        try:
            self.pi.set_PWM_dutycycle(self.enb, min(duty, 100))
            if speed > 0:
                self.pi.write(self.in3, 1)
                self.pi.write(self.in4, 0)
            elif speed < 0:
                self.pi.write(self.in3, 0)
                self.pi.write(self.in4, 1)
            else:
                self.pi.write(self.in3, 0)
                self.pi.write(self.in4, 0)
        except Exception as e:
            logger.error("pigpio motor_b error: %s", e)

    def set_speed(self, left, right):
        """set raw motor speeds.

        args:
            left: -100 to 100 (negative = backward)
            right: -100 to 100
        """
        left = max(-self.max_speed, min(self.max_speed, left))
        right = max(-self.max_speed, min(self.max_speed, right))
        with self._lock:
            self._set_motor_a(left)
            self._set_motor_b(right)
            self._current_speed = (left, right)

    def forward(self, speed=None):
        """move forward."""
        speed = speed or self.cruise_speed
        self.set_speed(speed, speed)
        log_event("motion", f"chassis forward at {speed}%")

    def backward(self, speed=None):
        """move backward."""
        speed = speed or self.cruise_speed
        self.set_speed(-speed, -speed)
        log_event("motion", f"chassis backward at {speed}%")

    def turn_left(self, speed=None):
        """turn left (right wheel faster)."""
        speed = speed or self.turn_speed
        self.set_speed(speed * 0.3, speed)
        log_event("motion", "chassis turning left")

    def turn_right(self, speed=None):
        """turn right (left wheel faster)."""
        speed = speed or self.turn_speed
        self.set_speed(speed, speed * 0.3)
        log_event("motion", "chassis turning right")

    def spin_left(self, speed=None):
        """spin in place to the left."""
        speed = speed or self.turn_speed
        self.set_speed(-speed, speed)
        log_event("motion", "chassis spinning left")

    def spin_right(self, speed=None):
        """spin in place to the right."""
        speed = speed or self.turn_speed
        self.set_speed(speed, -speed)
        log_event("motion", "chassis spinning right")

    def stop(self):
        """stop both motors immediately."""
        self._stop_event.set()
        self.set_speed(0, 0)
        log_event("motion", "chassis stopped")

    def brake(self):
        """stop motors abruptly by shorting terminals."""
        self._stop_event.set()
        with self._lock:
            try:
                self.pi.write(self.in1, 1)
                self.pi.write(self.in2, 1)
                self.pi.write(self.in3, 1)
                self.pi.write(self.in4, 1)
                self.pi.set_PWM_dutycycle(self.ena, 100)
                self.pi.set_PWM_dutycycle(self.enb, 100)
            except Exception as e:
                logger.error("pigpio brake error: %s", e)
            self._current_speed = (0, 0)
        log_event("motion", "chassis active brake applied")

    def move_for_duration(self, direction, speed, duration):
        """move in a direction for a set duration.

        args:
            direction: 'forward', 'backward', 'left', 'right'
            speed: motor speed percentage
            duration: time in seconds
        """
        actions = {
            "forward": self.forward,
            "backward": self.backward,
            "left": self.turn_left,
            "right": self.turn_right,
            "spin_left": self.spin_left,
            "spin_right": self.spin_right,
            "brake": self.brake,
        }
        action = actions.get(direction, self.stop)

        if action in (self.brake, self.stop):
            action()
            # clear the event set by brake/stop so we can wait
            self._stop_event.clear()
        else:
            with self._motion_lock:
                self._stop_event.clear()
                action(speed)
                
                # wait returns False if the timeout occurred, True if the event was set.
                if not self._stop_event.wait(duration):
                    self.stop()

    def move_to_angle(self, angle, duration=1.0):
        """move chassis to face a specific angle.

        approximates by spinning proportional to angle difference.

        args:
            angle: target angle in degrees (0-180, 90 = straight)
            duration: time to spend turning
        """
        self._stop_event.clear()
        interrupted = False
        if angle < 80:
            self.spin_right(self.turn_speed)
            interrupted = self._stop_event.wait(duration * (90 - angle) / 90)
        elif angle > 100:
            self.spin_left(self.turn_speed)
            interrupted = self._stop_event.wait(duration * (angle - 90) / 90)

        if not interrupted:
            self.set_speed(0, 0)
        log_event("motion", f"chassis aimed at ~{angle}°")

    def get_speed(self):
        """get current motor speeds."""
        with self._lock:
            return self._current_speed

    def cleanup(self):
        """stop motors and cleanup."""
        logger.info("cleaning up chassis controller")
        self.stop()
        with self._motion_lock:
            with self._lock:
                if _HAS_PIGPIO and hasattr(self.pi, 'connected'):
                    try:
                        self.pi.stop()
                    except Exception as e:
                        logger.error("pigpio stop error: %s", e)


if __name__ == "__main__":
    import yaml
    with open("config/simba_config.yaml") as f:
        config = yaml.safe_load(f)
    chassis = ChassisController(config)
    print("testing chassis...")
    chassis.forward(40)
    time.sleep(1)
    chassis.stop()
    time.sleep(0.5)
    chassis.spin_left(30)
    time.sleep(1)
    chassis.stop()
    chassis.cleanup()
    print("chassis test complete")
