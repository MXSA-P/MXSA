# _max_cyan_ — project_mxsa
"""imu reader — mpu6050 gyroscope/accelerometer via i2c."""

import time
import math
import threading

try:
    from smbus2 import SMBus
    _HAS_SMBUS = True
except ImportError:
    _HAS_SMBUS = False

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.motion.imu")


from typing import Dict, Any, Optional, Tuple

class IMUReader:
    """reads mpu6050 accelerometer + gyroscope data over i2c.

    mounted on the hand for orientation detection during grabs.
    """

    # class level lock for shared I2C bus
    _i2c_lock = threading.Lock()
    
    # mpu6050 registers
    PWR_MGMT_1 = 0x6b
    ACCEL_XOUT_H = 0x3b
    GYRO_XOUT_H = 0x43
    TEMP_OUT_H = 0x41
    ACCEL_CONFIG = 0x1c
    GYRO_CONFIG = 0x1b

    def __init__(self, config: Dict[str, Any]) -> None:
        imu_cfg = config["imu"]
        self.address = imu_cfg["i2c_address"]    # 0x68
        self.accel_range = imu_cfg["accel_range"]  # 2g
        self.gyro_range = imu_cfg["gyro_range"]    # 250 deg/s
        self.sample_rate = imu_cfg["sample_rate"]  # 50 hz
        self.cal_samples = imu_cfg["calibration_samples"]  # 100

        # calibration offsets
        self.accel_offset = [0.0, 0.0, 0.0]
        self.gyro_offset = [0.0, 0.0, 0.0]

        # scale factors
        accel_scales = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
        gyro_scales = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}
        self.accel_scale = max(1e-6, float(accel_scales.get(self.accel_range, 16384.0)))
        self.gyro_scale = max(1e-6, float(gyro_scales.get(self.gyro_range, 131.0)))

        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._last_data = None
        self._stop_event = threading.Event()

        if _HAS_SMBUS:
            try:
                self.bus = SMBus(1)
                self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0)
                time.sleep(0.1)
                logger.info("mpu6050 initialized on i2c bus 1")
            except Exception as e:
                logger.warning(f"mpu6050 init failed: {e}, using mock")
                self.bus = None
        else:
            logger.warning("smbus2 not available, using mock")
            self.bus = None

    def read_accel(self) -> Tuple[float, float, float]:
        """read accelerometer values in g.

        returns:
            tuple: (ax, ay, az) in g units
        """
        data = self.read_all()
        if not data:
            return (0.0, 0.0, 0.0)
        return (data["accel"]["x"], data["accel"]["y"], data["accel"]["z"])

    def read_gyro(self) -> Tuple[float, float, float]:
        """read gyroscope values in degrees/sec.

        returns:
            tuple: (gx, gy, gz) in deg/s
        """
        data = self.read_all()
        if not data:
            return (0.0, 0.0, 0.0)
        return (data["gyro"]["x"], data["gyro"]["y"], data["gyro"]["z"])

    def read_temp(self) -> float:
        """read temperature in celsius."""
        data = self.read_all()
        if not data:
            return 0.0
        return data["temp"]

    def read_all(self):
        """read all sensor data atomically."""
        if self.bus is None:
            return None
            
        with IMUReader._i2c_lock:
            try:
                # Read 14 bytes starting from ACCEL_XOUT_H (0x3B)
                # This ensures atomic snapshot of Accel (6), Temp (2), and Gyro (6)
                data = self.bus.read_i2c_block_data(self.address, self.ACCEL_XOUT_H, 14)
            except Exception as e:
                logger.error(f"mpu6050 read error: {e}")
                return None
                
        def _parse_16(high, low):
            val = (high << 8) | low
            return val - 0x10000 if val >= 0x8000 else val

        ax = _parse_16(data[0], data[1]) / self.accel_scale - self.accel_offset[0]
        ay = _parse_16(data[2], data[3]) / self.accel_scale - self.accel_offset[1]
        az = _parse_16(data[4], data[5]) / self.accel_scale - self.accel_offset[2]
        
        raw_temp = _parse_16(data[6], data[7])
        temp = (raw_temp / 340.0) + 36.53
        
        gx = _parse_16(data[8], data[9]) / self.gyro_scale - self.gyro_offset[0]
        gy = _parse_16(data[10], data[11]) / self.gyro_scale - self.gyro_offset[1]
        gz = _parse_16(data[12], data[13]) / self.gyro_scale - self.gyro_offset[2]

        accel = (ax, ay, az)
        gyro = (gx, gy, gz)

        parsed_data = {
            "accel": {"x": ax, "y": ay, "z": az},
            "gyro": {"x": gx, "y": gy, "z": gz},
            "temp": temp,
            "orientation": self.get_hand_orientation(accel),
            "tilt_angle": self.get_tilt_angle(accel, gyro),
            "is_falling": self.detect_fall(accel),
            "is_shaking": self.detect_shake(accel),
            "is_tapped": self.detect_tap(accel),
        }
        with self._lock:
            self._last_data = parsed_data
        return parsed_data

    def get_hand_orientation(self, accel=None):
        """determine hand orientation from accelerometer."""
        ax, ay, az = accel if accel else self.read_accel()

        if abs(az) > 0.8 and abs(ax) < 0.3 and abs(ay) < 0.3:
            return "level"
        elif ay > 0.5:
            return "tilted_right"
        elif ay < -0.5:
            return "tilted_left"
        elif ax > 0.5:
            return "pointing_down"
        elif ax < -0.5:
            return "pointing_up"
        return "level"

    def calibrate(self):
        """calibrate by averaging readings at rest."""
        logger.info(f"calibrating mpu6050 ({self.cal_samples} samples)...")
        accel_sum = [0.0, 0.0, 0.0]
        gyro_sum = [0.0, 0.0, 0.0]

        valid_samples = 0
        for _ in range(self.cal_samples):
            data = self.read_all()
            if data:
                accel_sum[0] += data["accel"]["x"]
                accel_sum[1] += data["accel"]["y"]
                accel_sum[2] += data["accel"]["z"]
                gyro_sum[0] += data["gyro"]["x"]
                gyro_sum[1] += data["gyro"]["y"]
                gyro_sum[2] += data["gyro"]["z"]
                valid_samples += 1
            if self._stop_event.wait(0.01):
                break

        n = max(1, valid_samples)
        self.accel_offset = [s / n for s in accel_sum]
        self.accel_offset[2] -= 1.0  # z should be 1g at rest
        self.gyro_offset = [s / n for s in gyro_sum]

        logger.info("mpu6050 calibration complete")
        log_event("imu", "calibration complete", {
            "accel_offset": self.accel_offset,
            "gyro_offset": self.gyro_offset,
        })

    def get_tilt_angle(self, accel=None, gyro=None):
        """get tilt angle from vertical in degrees using complementary filter."""
        # Note: caller should provide accel/gyro to avoid circular reads,
        # but fallback reads are possible if accessed directly.
        if accel is None or gyro is None:
            data = self.read_all()
            if not data:
                return 0.0
            return data["tilt_angle"]

        ax, ay, az = accel
        gx, gy, gz = gyro

        with self._lock:
            now = time.time()
            dt = now - getattr(self, '_last_time', now)
            self._last_time = now
            if dt > 1.0 or dt <= 0:
                dt = 0.02
    
            try:
                accel_pitch = math.degrees(math.atan2(
                    ay, math.sqrt(ax * ax + az * az)))
                accel_roll = math.degrees(math.atan2(-ax, az))
            except (ValueError, ZeroDivisionError):
                accel_pitch = 0.0
                accel_roll = 0.0
    
            if not hasattr(self, '_pitch'):
                self._pitch = accel_pitch
                self._roll = accel_roll
    
            alpha = 0.96
            self._pitch = alpha * (self._pitch + gx * dt) + \
                (1.0 - alpha) * accel_pitch
            self._roll = alpha * (self._roll + gy * dt) + \
                (1.0 - alpha) * accel_roll
    
            # Approximate tilt from vertical
            tilt = math.sqrt(self._pitch**2 + self._roll**2)
            
        return round(tilt, 1)

    def detect_fall(self, accel=None):
        """detect if the device is falling (freefall)."""
        ax, ay, az = accel if accel else self.read_accel()
        magnitude = math.sqrt(ax * ax + ay * ay + az * az)
        return magnitude < 0.3

    def detect_shake(self, accel=None):
        """detect rapid shaking."""
        ax, ay, az = accel if accel else self.read_accel()
        magnitude = math.sqrt(ax * ax + ay * ay + (az - 1.0)**2)
        return magnitude > 1.5

    def detect_tap(self, accel=None):
        """detect a sharp tap."""
        ax, ay, az = accel if accel else self.read_accel()
        accel_mag = math.sqrt(ax * ax + ay * ay + az * az)
        return 1.5 < accel_mag < 3.0

    def is_moving(self, gyro=None):
        """detect if hand is in motion based on gyro readings."""
        gx, gy, gz = gyro if gyro else self.read_gyro()
        magnitude = math.sqrt(gx * gx + gy * gy + gz * gz)
        return magnitude > 10.0  # threshold in deg/s

    def start_continuous(self, callback=None):
        """start continuous reading in background thread.

        args:
            callback: function(data_dict) called on each read
        """
        if not self._stop_event.is_set() and self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()

        def _reader():
            while not self._stop_event.is_set():
                data = self.read_all()
                if callback:
                    callback(data)
                self._stop_event.wait(1.0 / max(0.1, self.sample_rate))

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()
        logger.info(f"imu continuous reading started at {self.sample_rate}hz")

    def stop_continuous(self):
        """stop continuous reading."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        logger.info("imu continuous reading stopped")

    def get_last_data(self):
        """get last read data from continuous mode."""
        with self._lock:
            return self._last_data

    def cleanup(self):
        """cleanup i2c resources."""
        self.stop_continuous()
        self._stop_event.set()
        if self.bus:
            try:
                with IMUReader._i2c_lock:
                    self.bus.close()
            except Exception:
                pass
        logger.info("imu cleanup complete")


if __name__ == "__main__":
    import yaml
    with open("config/simba_config.yaml") as f:
        config = yaml.safe_load(f)
    imu = IMUReader(config)
    imu.calibrate()
    for _ in range(10):
        data = imu.read_all()
        print(
            f"orientation: {
                data['orientation']}, tilt: {
                data['tilt_angle']}°")
        time.sleep(0.2)
    imu.cleanup()
