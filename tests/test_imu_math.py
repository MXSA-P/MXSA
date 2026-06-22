# _max_cyan_ — project_mxsa
import pytest
import time
from simba.motion.imu import IMUReader


@pytest.fixture
def mock_config():
    return {
        "imu": {
            "i2c_address": 0x68,
            "accel_range": 2,
            "gyro_range": 250,
            "sample_rate": 50,
            "calibration_samples": 10
        }
    }


def test_get_tilt_angle(mock_config):
    imu = IMUReader(mock_config)

    # For a level tilt, ax=0, ay=0, az=1.0.
    # For 45 deg pitch, ax ~0.707, az ~0.707
    accel = (0.7071, 0.0, 0.7071)
    gyro = (0.0, 0.0, 0.0)

    # Needs to be called twice to establish dt
    imu.get_tilt_angle(accel=accel, gyro=gyro)
    time.sleep(0.02)
    tilt = imu.get_tilt_angle(accel=accel, gyro=gyro)

    # Roll is 0, Pitch is ~45 degrees -> tilt is ~45
    assert 44.0 <= tilt <= 46.0


def test_complementary_filter(mock_config):
    imu = IMUReader(mock_config)

    # Initially 0 tilt
    tilt = imu.get_tilt_angle(accel=(0.0, 0.0, 1.0), gyro=(0.0, 0.0, 0.0))
    assert tilt == 0.0

    # Sudden gyro spike
    time.sleep(0.1)  # dt = 0.1s => gyro change ~10 degrees
    tilt = imu.get_tilt_angle(accel=(0.0, 0.0, 1.0), gyro=(100.0, 0.0, 0.0))

    # Since alpha is 0.96, the tilt should start to follow the gyro
    assert tilt > 0.0
