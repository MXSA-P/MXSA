import subprocess
import io
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from PIL import Image, ImageDraw
import numpy as np

import sys
import os
# Ensure the root directory is in the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load Simba AI Vision Pipeline
from simba.vision.hybrid_detector import HybridDetector
print("Initializing HybridDetector (YOLO + SVM)...")
detector = HybridDetector()

HOST = "0.0.0.0"
PORT = 8081  # Port 8081 to avoid conflict with Simba's main server (8080)
IMAGE = "capture.jpg"

class Handler(BaseHTTPRequestHandler):
    def do_GET_image(self):
        pass

    def do_GET(self):
        if self.path == "/image":
            # 1. Capture image using user's rpicam-still approach
            print("\nCapturing frame with rpicam-still...")
            subprocess.run([
                "rpicam-still",
                "-o", IMAGE,
                "--nopreview",
                "-t", "1000",   # 1 second warmup
                "--width", "640",
                "--height", "480"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            img_bytes = b""
            try:
                # 2. Open with PIL and convert to AI-compatible numpy array
                img = Image.open(IMAGE).convert("RGB")
                frame_np = np.array(img)
                
                # 3. Run AI Detection!
                print("Running AI object detection...")
                detections = detector.detect_objects(frame_np)
                print(f"Detected {len(detections)} objects: {detections}")
                
                # 4. Draw bounding boxes so we can literally SEE what the AI sees
                draw = ImageDraw.Draw(img)
                for det in detections:
                    box = det.get("box")
                    label = det.get("label", "unknown")
                    conf = det.get("confidence", 0.0)
                    if box and len(box) == 4:
                        x1, y1, x2, y2 = box
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                        draw.text((x1, y1 - 15), f"{label} ({conf*100:.0f}%)", fill="cyan")
                
                # Save annotated image to memory
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                img_bytes = buf.getvalue()
                
            except Exception as e:
                print(f"AI Detection failed: {e}")
                with open(IMAGE, "rb") as f:
                    img_bytes = f.read()

            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.end_headers()
            self.wfile.write(img_bytes)
            
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
<html>
<head>
<meta http-equiv="refresh" content="2">
<title>Simba AI Vision Diagnostic</title>
</head>
<body style="background: #111; color: white; font-family: sans-serif; text-align: center;">
<h1>Simba AI Vision Diagnostic</h1>
<p>Capturing using rpicam-still and mapping AI Tensor bounds...</p>
<img src="/image" width="640" style="border: 3px solid #333; border-radius: 10px;">
</body>
</html>
""")

if __name__ == "__main__":
    print("=====================================")
    print("  AI VISION HARDWARE TESTER STARTED  ")
    print(f"  Go to http://{HOST}:{PORT}        ")
    print("=====================================")
    HTTPServer((HOST, PORT), Handler).serve_forever()
