import re

with open("simba/motion/arm.py", "r") as f:
    content = f.read()

# Find the move_smooth function and fix its indentation
pattern = r"""(    def move_smooth\(self, target_angles, speed=None\):
        \"\"\"Move all arm servos simultaneously to target angles\.\"\"\"
        if speed is None:
            speed = self\.move_speed
        speed = max\(0\.1, speed\)

        with self\._motion_lock:
            with self\._lock:
                self\._moving = True
                start_angles = \{k: self\.current\[k\] for k in self\.current\}

)        targets = \{
            "rotation": max\(
                self\.rotation_min, min\(
                    self\.rotation_max, target_angles\.get\(
                        "rotation", start_angles\["rotation"\]\)\)\), "elbow": max\(
                self\.elbow_min, min\(
                    self\.elbow_max, target_angles\.get\(
                        "elbow", start_angles\["elbow"\]\)\)\), "elbow_2": max\(
                self\.elbow_2_min, min\(
                    self\.elbow_2_max, target_angles\.get\(
                        "elbow_2", start_angles\["elbow_2"\]\)\)\), "wrist": max\(
                self\.wrist_min, min\(
                    self\.wrist_max, target_angles\.get\(
                        "wrist", start_angles\["wrist"\]\)\)\), \}

        steps = max\(
            abs\(targets\["rotation"\] - start_angles\["rotation"\]\),
            abs\(targets\["elbow"\] - start_angles\["elbow"\]\),
            abs\(targets\["elbow_2"\] - start_angles\["elbow_2"\]\),
            abs\(targets\["wrist"\] - start_angles\["wrist"\]\),
        \)
        num_steps = max\(1, int\(steps / speed\)\)

        interrupted = False
        for i in range\(1, num_steps \+ 1\):
            t = i / num_steps
            ease_t = 6 \* \(t \*\* 5\) - 15 \* \(t \*\* 4\) \+ 10 \* \(t \*\* 3\)
            with self\._lock:
                for key, pin in \[\("rotation", self\.rotation_pin\),
                                 \("elbow", self\.elbow_pin\),
                                 \("elbow_2", self\.elbow_2_pin\),
                                 \("wrist", self\.wrist_pin\)\]:
                    angle = start_angles\[key\] \+ ease_t \* \\
                        \(targets\[key\] - start_angles\[key\]\)
                    self\._set_servo\(pin, angle\)
                    self\.current\[key\] = angle

            if self\._stop_event\.wait\(0\.02\):
                interrupted = True
                with self\._lock:
                    for key in targets:
                        self\.current\[key\] = start_angles\[key\] \+ \\
                            ease_t \* \(targets\[key\] - start_angles\[key\]\)
                    self\._moving = False
                return

        # set final positions
        with self\._lock:
            if not interrupted:
                for key in targets:
                    self\.current\[key\] = targets\[key\]
            self\._moving = False"""

replacement = """    def move_smooth(self, target_angles, speed=None):
        \"\"\"Move all arm servos simultaneously to target angles.\"\"\"
        if speed is None:
            speed = self.move_speed
        speed = max(0.1, speed)

        with self._motion_lock:
            with self._lock:
                self._moving = True
                start_angles = {k: self.current[k] for k in self.current}

            targets = {
                "rotation": max(self.rotation_min, min(self.rotation_max, target_angles.get("rotation", start_angles["rotation"]))),
                "elbow": max(self.elbow_min, min(self.elbow_max, target_angles.get("elbow", start_angles["elbow"]))),
                "elbow_2": max(self.elbow_2_min, min(self.elbow_2_max, target_angles.get("elbow_2", start_angles["elbow_2"]))),
                "wrist": max(self.wrist_min, min(self.wrist_max, target_angles.get("wrist", start_angles["wrist"])))
            }

            steps = max(
                abs(targets["rotation"] - start_angles["rotation"]),
                abs(targets["elbow"] - start_angles["elbow"]),
                abs(targets["elbow_2"] - start_angles["elbow_2"]),
                abs(targets["wrist"] - start_angles["wrist"])
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
                        angle = start_angles[key] + ease_t * (targets[key] - start_angles[key])
                        self._set_servo(pin, angle)
                        self.current[key] = angle

                if self._stop_event.wait(0.02):
                    interrupted = True
                    with self._lock:
                        for key in targets:
                            self.current[key] = start_angles[key] + ease_t * (targets[key] - start_angles[key])
                        self._moving = False
                    return

            # set final positions
            with self._lock:
                if not interrupted:
                    for key in targets:
                        self.current[key] = targets[key]
                self._moving = False"""

if re.search(pattern, content):
    new_content = re.sub(pattern, replacement, content)
    with open("simba/motion/arm.py", "w") as f:
        f.write(new_content)
    print("Fixed move_smooth indentation")
else:
    print("Pattern not found!")
