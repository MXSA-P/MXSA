# _max_cyan_ — project_mxsa
"""imu reader — mpu6050 gyroscope/accelerometer via i2c."""

import math
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from smbus2 import SMBus
    _HAS_SMBUS = True
except ImportError:
    _HAS_SMBUS = False

from simba.utils.logger import get_logger, log_event

logger = get_logger("simba.motion.imu")

_UNPACK_7H = struct.Struct('>7h')

def _run_with_timeout(func: Callable[..., Any], *args: Any, timeout: float = 0.5) -> Any:
    """Run a function in a thread with a strict timeout boundary."""
    result: List[Any] = []
    exception: List[Exception] = []

    def worker() -> None:
        try:
            result.append(func(*args))
        except Exception as e:
            exception.append(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
    if exception:
        raise exception[0]
    if result:
        return result[0]
    return None

class IMUReader:
    """reads mpu6050 accelerometer + gyroscope data over i2c.

    mounted on the hand for orientation detection during grabs.
    """

    # class level lock for shared I2C bus
    _i2c_lock: threading.Lock = threading.Lock()
    
    # mpu6050 registers
    PWR_MGMT_1: int = 0x6b
    ACCEL_XOUT_H: int = 0x3b
    GYRO_XOUT_H: int = 0x43
    TEMP_OUT_H: int = 0x41
    ACCEL_CONFIG: int = 0x1c
    GYRO_CONFIG: int = 0x1b

    def __init__(self, config: Dict[str, Any]) -> None:
        """initialize the IMU reader with given config."""
        imu_cfg = config["imu"]
        self.address = imu_cfg["i2c_address"]    # 0x68
        self.accel_range = imu_cfg["accel_range"]  # 2g
        self.gyro_range = imu_cfg["gyro_range"]    # 250 deg/s
        self.sample_rate = imu_cfg["sample_rate"]  # 50 hz
        self.cal_samples = imu_cfg["calibration_samples"]  # 100

        # calibration offsets
        self.accel_offset: List[float] = [0.0, 0.0, 0.0]
        self.gyro_offset: List[float] = [0.0, 0.0, 0.0]

        # scale factors
        accel_scales: Dict[int, float] = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
        gyro_scales: Dict[int, float] = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}
        self.accel_scale: float = max(1e-6, float(accel_scales.get(self.accel_range, 16384.0)))
        self.gyro_scale: float = max(1e-6, float(gyro_scales.get(self.gyro_range, 131.0)))

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()
        self._last_data: Optional[Dict[str, Any]] = None
        self._stop_event: threading.Event = threading.Event()
        self._connected: bool = True
        self.bus: Any = None
        
        # Tilt state
        self._pitch: Optional[float] = None
        self._roll: Optional[float] = None
        self._last_time: float = time.time()

        if _HAS_SMBUS:
            try:
                self.bus = SMBus(1)
                _run_with_timeout(
                    self.bus.write_byte_data, self.address, self.PWR_MGMT_1, 0, timeout=0.5
                )
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

    def read_all(self) -> Optional[Dict[str, Any]]:
        """read all sensor data atomically."""
        if self.bus is None:
            return None
            
        with IMUReader._i2c_lock:
            try:
                if not getattr(self, '_connected', True):
                    _run_with_timeout(
                        self.bus.write_byte_data, self.address, self.PWR_MGMT_1, 0, timeout=0.2
                    )
                    
                # Read 14 bytes starting from ACCEL_XOUT_H (0x3B)
                # This ensures atomic snapshot of Accel (6), Temp (2), and Gyro (6)
                data = _run_with_timeout(
                    self.bus.read_i2c_block_data, self.address, self.ACCEL_XOUT_H, 14, timeout=0.2
                )
                
                if not getattr(self, '_connected', True):
                    logger.info("mpu6050 reconnected")
                    self._connected = True
            except Exception as e:
                if getattr(self, '_connected', True):
                    logger.error(f"mpu6050 read error: {e}")
                    self._connected = False
                return None
                
        if not data or len(data) != 14:
            logger.error(f"mpu6050 read error: malformed data length {len(data) if data else 0}")
            return None
            
        try:
            # Unpack 14 bytes into 7 big-endian signed 16-bit integers
            ax_raw, ay_raw, az_raw, raw_temp, gx_raw, gy_raw, gz_raw = (
                _UNPACK_7H.unpack(bytes(data))
            )
        except (struct.error, ValueError) as e:
            logger.error(f"mpu6050 unpack error: {e}")
            return None

        ax = ax_raw / self.accel_scale - self.accel_offset[0]
        ay = ay_raw / self.accel_scale - self.accel_offset[1]
        az = az_raw / self.accel_scale - self.accel_offset[2]
        
        temp = (raw_temp / 340.0) + 36.53
        
        gx = gx_raw / self.gyro_scale - self.gyro_offset[0]
        gy = gy_raw / self.gyro_scale - self.gyro_offset[1]
        gz = gz_raw / self.gyro_scale - self.gyro_offset[2]

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

    def get_hand_orientation(self, accel: Optional[Tuple[float, float, float]] = None) -> str:
        """determine hand orientation from accelerometer."""
        ax, ay, az = accel if accel is not None else self.read_accel()

        if abs(ay) > 0.8 and abs(ax) < 0.5 and abs(az) < 0.5:
            return "level"
        elif ax < -0.5:
            return "tilted_right"
        elif ax > 0.5:
            return "tilted_left"
        elif az > 0.5:
            return "pointing_up"
        elif az < -0.5:
            return "pointing_down"
        return "level"

    def calibrate(self) -> None:
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

        if valid_samples == 0:
            logger.warning("mpu6050 calibration failed: no valid samples")
            return

        self.accel_offset[0] += accel_sum[0] / valid_samples
        self.accel_offset[1] += accel_sum[1] / valid_samples
        self.accel_offset[2] += (accel_sum[2] / valid_samples) - 1.0  # z should be 1g at rest
        self.gyro_offset[0] += gyro_sum[0] / valid_samples
        self.gyro_offset[1] += gyro_sum[1] / valid_samples
        self.gyro_offset[2] += gyro_sum[2] / valid_samples

        logger.info("mpu6050 calibration complete")
        log_event("imu", "calibration complete", {
            "accel_offset": self.accel_offset,
            "gyro_offset": self.gyro_offset,
        })

    def get_tilt_angle(
        self,
        accel: Optional[Tuple[float, float, float]] = None,
        gyro: Optional[Tuple[float, float, float]] = None
    ) -> float:
        """get tilt angle from vertical in degrees using complementary filter."""
        # Note: caller should provide accel/gyro to avoid circular reads,
        # but fallback reads are possible if accessed directly.
        if accel is None or gyro is None:
            data = self.read_all()
            if not data:
                return 0.0
            return data["tilt_angle"] if data else 0.0

        ax, ay, az = accel
        gx, gy, gz = gyro

        with self._lock:
            now = time.time()
            dt = now - self._last_time
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
    
            if self._pitch is None or self._roll is None:
                self._pitch = accel_pitch
                self._roll = accel_roll
    
            alpha = 0.96
            self._pitch = alpha * (self._pitch + gx * dt) + \
                (1.0 - alpha) * accel_pitch
            self._roll = alpha * (self._roll + gy * dt) + \
                (1.0 - alpha) * accel_roll
    
            # Approximate tilt from vertical
            tilt = math.sqrt(self._pitch ** 2 + self._roll ** 2)
            
        return round(tilt, 1)

    def detect_fall(self, accel: Optional[Tuple[float, float, float]] = None) -> bool:
        """detect if the device is falling (freefall)."""
        ax, ay, az = accel if accel is not None else self.read_accel()
        magnitude = math.sqrt(ax * ax + ay * ay + az * az)
        return magnitude < 0.3

    def detect_shake(self, accel: Optional[Tuple[float, float, float]] = None) -> bool:
        """detect rapid shaking."""
        ax, ay, az = accel if accel is not None else self.read_accel()
        magnitude = math.sqrt(ax * ax + ay * ay + (az - 1.0) ** 2)
        return magnitude > 1.5

    def detect_tap(self, accel: Optional[Tuple[float, float, float]] = None) -> bool:
        """detect a sharp tap."""
        ax, ay, az = accel if accel is not None else self.read_accel()
        accel_mag = math.sqrt(ax * ax + ay * ay + az * az)
        return 1.5 < accel_mag < 3.0

    def is_moving(self, gyro: Optional[Tuple[float, float, float]] = None) -> bool:
        """detect if hand is in motion based on gyro readings."""
        gx, gy, gz = gyro if gyro is not None else self.read_gyro()
        magnitude = math.sqrt(gx * gx + gy * gy + gz * gz)
        return magnitude > 10.0  # threshold in deg/s

    def start_continuous(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
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
                if data is not None:
                    if callback:
                        callback(data)
                    wait_time = 1.0 / max(0.1, self.sample_rate)
                else:
                    # Sleep longer if disconnected to avoid spamming the bus and consuming CPU
                    wait_time = 1.0
                self._stop_event.wait(wait_time)

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()
        logger.info(f"imu continuous reading started at {self.sample_rate}hz")

    def stop_continuous(self) -> None:
        """stop continuous reading."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        logger.info("imu continuous reading stopped")

    def get_last_data(self) -> Optional[Dict[str, Any]]:
        """get last read data from continuous mode."""
        with self._lock:
            return self._last_data

    def cleanup(self) -> None:
        """cleanup i2c resources."""
        self.stop_continuous()
        self._stop_event.set()
        if self.bus:
            try:
                with IMUReader._i2c_lock:
                    _run_with_timeout(self.bus.close, timeout=0.5)
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
        orientation = data['orientation'] if data else 'mock_orientation'
        tilt = data['tilt_angle'] if data else 0.0
        print(f"orientation: {orientation}, tilt: {tilt}°")
        time.sleep(0.2)
    imu.cleanup()
