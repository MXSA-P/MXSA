# SIMBA BIONIC ARM ROBOT — HARDWARE CONNECTIONS GUIDE

## Easy Pinout Summary

**Motor Driver (L298N 6-pin)**
*   ENA -> GPIO 12 (Motor A PWM Speed)
*   IN1 -> GPIO 5 (Motor A Forward: High, Reverse: Low)
*   IN2 -> GPIO 6 (Motor A Reverse: High, Forward: Low)
*   IN3 -> GPIO 16 (Motor B Forward: High, Reverse: Low)
*   IN4 -> GPIO 26 (Motor B Reverse: High, Forward: Low)
*   ENB -> GPIO 13 (Motor B PWM Speed)

**Logic Table: Motor A (Left Wheel)**
|    State    | IN1  | IN2  | ENA |
|     :---    | :--- | :--- | :---|
| **Forward** | `1`  | `0`  | PWM |
| **Reverse** | `0`  | `1`  | PWM |
| **Stop**    | `0`  | `0`  | `0` |
| **Brake**   | `1`  | `1`  |`255`|

**Logic Table: Motor B (Right Wheel)**
| State | IN3 (GPIO 16) | IN4 (GPIO 26) | ENB (GPIO 13) |
| :--- | :--- | :--- | :--- |
| **Forward** | `1` (HIGH) | `0` (LOW) | PWM |
| **Reverse** | `0` (LOW) | `1` (HIGH) | PWM |
| **Stop** | `0` (LOW) | `0` (LOW) | `0` |
| **Brake** | `1` (HIGH) | `1` (HIGH) | `255` |

**Arm Servos (7x SG90)**
*   Arm Rotation -> GPIO 22
*   Arm Elbow 1 -> GPIO 23
*   Arm Elbow 2 -> GPIO 25
*   Arm Wrist -> GPIO 24
*   Finger 1 -> GPIO 4
*   Finger 2 -> GPIO 17
*   Finger 3 -> GPIO 27

**Sensors & Audio**
*   MPU6050 SDA -> GPIO 2
*   MPU6050 SCL -> GPIO 3
*   Camera -> CSI Port Ribbonmmn.

**INMP441 MEMS Microphone (I2S)**
| INMP441 Pin | Pi GPIO | Description |
| :--- | :--- | :---    |
| **GND**     |  GND    |         Ground 
| **VDD**     | 3.3V    | Power (3.3V only!) 
| **SD**      | GPIO 20 | Serial Data Out 
| **L/R**     | GND     | Low = Left channel (mono) 
| **WS**      | GPIO 19 | Left/Right Clock (LRCLK) 
| **SCK**     | GPIO 18 | Bit Clock (BCLK) 

---

## Power Connections

*   **Batteries:** 2x 18650 parallel pack (3.7V) powers the TP4056, L298N VCC terminal, and the massive 7-Servo Power Rail.
*   **Raspberry Pi:** Powered by 5V Powerbank.
*   **IMPORTANT:** You must connect the Pi GND to the Battery/L298N GND to share a common ground, otherwise signals will float.

---

## Auto-Return Navigation

Simba records every movement command (direction, speed, and duration) since boot, building an in-memory breadcrumb trail of its path. When either of the following conditions is met, the auto-return sequence activates:

1. **Voice trigger** — The user says **"go charge"**.
2. **Time trigger** — **30 minutes** have elapsed since the last charge cycle ended.

**Return procedure:**
*   Simba stops its current activity and replays the recorded movement list **in reverse order**, with each movement's direction inverted (forward ↔ reverse, left ↔ right), effectively retracing its path back to the charging station.
*   Once it reaches the station, it enters **charging mode** and remains stationary.

**Resume procedure:**
*   After **15 minutes** of charging, Simba automatically exits charging mode and resumes its normal roaming behaviour.
*   The movement history is cleared, and a new breadcrumb trail begins recording from that point.

_max_cyan_ — project_mxsa
