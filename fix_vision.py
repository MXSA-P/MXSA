import os
import sys

def replace_in_file(path, old, new):
    with open(path, 'r') as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(path, 'w') as f:
            f.write(content)
        print(f"Updated {path}")
    else:
        print(f"Could not find target in {path}")

# Fix hybrid_detector.py
replace_in_file('simba/vision/hybrid_detector.py',
    '"box": [w * 0.1, h * 0.1, w * 0.9, h * 0.9]',
    '"bbox": [int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8)]')

replace_in_file('simba/vision/hybrid_detector.py',
    'return True, svm_conf, [w * 0.1, h * 0.1, w * 0.9, h * 0.9]',
    'return True, svm_conf, [int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8)]')

# Fix yolo_detector.py
replace_in_file('simba/vision/yolo_detector.py',
    'if self.model is None or frame_np is None:',
    "if self.model is None or frame_np is None or not hasattr(frame_np, 'shape') or len(frame_np.shape) < 2:")

# Fix detector.py
replace_in_file('simba/vision/detector.py',
    "if frame is None or not hasattr(frame, 'shape'):",
    "if frame is None or not hasattr(frame, 'shape') or len(frame.shape) < 2:")

replace_in_file('simba/vision/detector.py',
    'resized = self._resize_frame(frame, self.input_size)\n\n        # prepare input tensor',
    '''resized = self._resize_frame(frame, self.input_size)
        if len(resized.shape) == 2:
            resized = np.stack((resized,) * 3, axis=-1)
        elif resized.shape[2] == 4:
            resized = resized[..., :3]

        # prepare input tensor''')

replace_in_file('simba/vision/detector.py',
    'gc.collect()',
    '# gc.collect() removed for performance')

replace_in_file('simba/vision/detector.py',
    '''        try:
            from simba.vision.yolo_detector import YoloDetector
            self.yolo = YoloDetector()
        except ImportError:
            self.yolo = None''',
    '''        try:
            from simba.vision.yolo_detector import YoloDetector
            self.yolo = YoloDetector()
        except Exception as e:
            logger.warning(f"Failed to load YoloDetector: {e}")
            self.yolo = None''')

# Fix scanner.py
replace_in_file('simba/vision/scanner.py',
    'width = box[3] - box[1]  # xmax - xmin',
    'width = box[2] - box[0]  # xmax - xmin')

replace_in_file('simba/vision/scanner.py',
    '''        self.angles: List[int] = [
            int(a) for a in ai_cfg.get(
                "scan_sweep_angles", [
                    0, 45, 90, 135, 180])]''',
    '''        angles_raw = ai_cfg.get("scan_sweep_angles", [0, 45, 90, 135, 180])
        if isinstance(angles_raw, str):
            angles_raw = [int(x.strip()) for x in angles_raw.split(',')]
        self.angles: List[int] = [int(a) for a in angles_raw]''')

