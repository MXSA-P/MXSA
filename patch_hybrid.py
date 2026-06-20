import re

path = "/home/mazo/Documents/MXSA/simba/vision/hybrid_detector.py"
with open(path, "r") as f:
    content = f.read()

# Add imports
content = content.replace("import numpy as np", "import numpy as np\nimport concurrent.futures\nimport math")

# Add executor in __init__
init_replacement = """    def __init__(self) -> None:
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.svm = ObjectDetector()"""
content = content.replace("    def __init__(self) -> None:\n        self.svm = ObjectDetector()", init_replacement)

# Add __del__
del_method = """
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    def detect_objects"""
content = content.replace("    def detect_objects", del_method)

# Fix detect_objects
detect_obj_old = """        # 1. YOLO detections
        if self.yolo:
            try:
                detections.extend(self.yolo.detect_objects(frame_np))
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")

        # 2. Custom SVM detection (fallback for custom objects)
        svm_label, svm_conf = self.svm.classify(frame_np)

        # if SVM found something custom that YOLO didn't
        if svm_label != "unknown" and svm_conf > 0.5:"""

detect_obj_new = """        yolo_future = None
        if self.yolo:
            yolo_future = self.executor.submit(self.yolo.detect_objects, frame_np)
            
        svm_future = self.executor.submit(self.svm.classify, frame_np)

        if yolo_future:
            try:
                yolo_dets = yolo_future.result()
                for d in yolo_dets:
                    if math.isnan(d.get("confidence", 0.0)):
                        d["confidence"] = 0.0
                    d["confidence"] = max(0.0, min(1.0, float(d["confidence"])))
                detections.extend(yolo_dets)
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")

        try:
            svm_label, svm_conf = svm_future.result()
        except Exception as e:
            logger.error(f"SVM classification failed: {e}")
            svm_label, svm_conf = "unknown", 0.0
            
        if math.isnan(svm_conf):
            svm_conf = 0.0
        svm_conf = max(0.0, min(1.0, float(svm_conf)))

        # if SVM found something custom that YOLO didn't
        if svm_label != "unknown" and svm_conf > 0.5:"""
content = content.replace(detect_obj_old, detect_obj_new)

# Fix is_object_in_frame
is_obj_old = """        # check YOLO first
        if self.yolo:
            try:
                found, conf, bbox = self.yolo.is_object_in_frame(
                    frame_np, target_label, threshold)
                if found:
                    return found, conf, bbox
            except Exception as e:
                logger.error(f"YOLO is_object_in_frame failed: {e}")

        # check SVM
        svm_label, svm_conf = self.svm.classify(frame_np)
        if svm_label == target_label and svm_conf >= threshold:
            h, w = frame_np.shape[:2]
            return True, svm_conf, [int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8)]

        return False, 0.0, None"""

is_obj_new = """        yolo_future = None
        if self.yolo:
            yolo_future = self.executor.submit(self.yolo.is_object_in_frame, frame_np, target_label, threshold)
            
        svm_future = self.executor.submit(self.svm.classify, frame_np)

        if yolo_future:
            try:
                found, conf, bbox = yolo_future.result()
                if found:
                    if math.isnan(conf): conf = 0.0
                    conf = max(0.0, min(1.0, float(conf)))
                    return found, conf, bbox
            except Exception as e:
                logger.error(f"YOLO is_object_in_frame failed: {e}")

        try:
            svm_label, svm_conf = svm_future.result()
            if math.isnan(svm_conf): svm_conf = 0.0
            svm_conf = max(0.0, min(1.0, float(svm_conf)))
            if svm_label == target_label and svm_conf >= threshold:
                h, w = frame_np.shape[:2]
                return True, svm_conf, [int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8)]
        except Exception as e:
            logger.error(f"SVM is_object_in_frame failed: {e}")

        return False, 0.0, None"""
content = content.replace(is_obj_old, is_obj_new)

# Fix classify
classify_old = """        # check YOLO
        if self.yolo:
            try:
                yolo_label, yolo_conf = self.yolo.classify(frame_np)
                if yolo_conf > best_conf:
                    best_label = yolo_label
                    best_conf = yolo_conf
            except Exception as e:
                logger.error(f"YOLO classify failed: {e}")

        # check SVM
        svm_label, svm_conf = self.svm.classify(frame_np)
        if svm_conf > best_conf:
            best_label = svm_label
            best_conf = svm_conf

        return best_label, best_conf"""

classify_new = """        yolo_future = None
        if self.yolo:
            yolo_future = self.executor.submit(self.yolo.classify, frame_np)
            
        svm_future = self.executor.submit(self.svm.classify, frame_np)

        if yolo_future:
            try:
                yolo_label, yolo_conf = yolo_future.result()
                if math.isnan(yolo_conf): yolo_conf = 0.0
                yolo_conf = max(0.0, min(1.0, float(yolo_conf)))
                if yolo_conf > best_conf:
                    best_label = yolo_label
                    best_conf = yolo_conf
            except Exception as e:
                logger.error(f"YOLO classify failed: {e}")

        try:
            svm_label, svm_conf = svm_future.result()
            if math.isnan(svm_conf): svm_conf = 0.0
            svm_conf = max(0.0, min(1.0, float(svm_conf)))
            if svm_conf > best_conf:
                best_label = svm_label
                best_conf = svm_conf
        except Exception as e:
            logger.error(f"SVM classify failed: {e}")

        return best_label, best_conf"""
content = content.replace(classify_old, classify_new)

with open(path, "w") as f:
    f.write(content)
