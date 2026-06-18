import math

def move_to_xyz(x: float, y: float, z: float) -> None:
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
        scale = (L1 + L2 - 0.1) / target_dist
        r *= scale
        z *= scale

    # calculate wrist angle (theta2) using cosine rule
    c2 = (r**2 + z**2 - L1**2 - L2**2) / (2 * L1 * L2)
    # clamp c2 to [-1, 1] to prevent domain errors if target is unreachable
    c2 = max(-1.0, min(1.0, c2))
    theta2_rad = math.acos(c2)

    # calculate elbow angle (theta1)
    k1 = L1 + L2 * c2
    k2 = L2 * math.sin(theta2_rad)
    theta1_rad = math.atan2(z, r) - math.atan2(k2, k1)

    elbow_angle = 180 - math.degrees(theta1_rad)
    elbow_2_angle = 90 + math.degrees(theta2_rad)

import random
for _ in range(100000):
    x = random.uniform(-100, 100)
    y = random.uniform(-100, 100)
    z = random.uniform(-100, 100)
    move_to_xyz(x, y, z)

for _ in range(100000):
    x = random.uniform(-0.1, 0.1)
    y = random.uniform(-0.1, 0.1)
    z = random.uniform(-0.1, 0.1)
    move_to_xyz(x, y, z)

print("Done")
