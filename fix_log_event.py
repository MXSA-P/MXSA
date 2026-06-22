import re

with open("simba/motion/arm.py", "r") as f:
    content = f.read()

pattern = r"""        log_event\(
            "motion", f"IK calculated: rot=\{
                rot_angle:\.1f\}, elbow=\{
                elbow_angle:\.1f\}, elbow_2=\{
                elbow_2_angle:\.1f\}, wrist=\{
                wrist_angle:\.1f\} for xyz\(\{x\},\{y\},\{z\}\)"\)"""

replacement = """        log_event("motion", f"IK calculated: rot={rot_angle:.1f}, elbow={elbow_angle:.1f}, elbow_2={elbow_2_angle:.1f}, wrist={wrist_angle:.1f} for xyz({x},{y},{z})")"""

if re.search(pattern, content):
    new_content = re.sub(pattern, replacement, content)
    with open("simba/motion/arm.py", "w") as f:
        f.write(new_content)
    print("Fixed log_event formatting")
else:
    print("Pattern not found!")
