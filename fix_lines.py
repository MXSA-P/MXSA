with open('simba/vision/detector.py', 'r') as f:
    content = f.read()

content = content.replace(
    'if "delegate" in err_str or "unresolved custom op" in err_str or "custom op" in err_str:',
    'if ("delegate" in err_str or "unresolved custom op" in err_str\n                        or "custom op" in err_str):'
)

content = content.replace(
    '"classifier not found (expected on fresh install, falling back to YOLO): %s", self._classifier_path)',
    '"classifier not found (expected on fresh install, falling back "\n                "to YOLO): %s", self._classifier_path)'
)

with open('simba/vision/detector.py', 'w') as f:
    f.write(content)
