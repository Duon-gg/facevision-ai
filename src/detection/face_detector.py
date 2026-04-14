"""
detection/face_detector.py - Phát hiện khuôn mặt (Computer Vision)
MTCNN (primary) + Haar Cascade (fallback)
MTCNN: Multi-task Cascaded CNN - 3 mạng P-Net → R-Net → O-Net
"""

import cv2
import torch
import numpy as np
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


class FaceDetector:
    """
    Phát hiện khuôn mặt:
    - Primary: MTCNN (chính xác cao, hỗ trợ góc nghiêng, ánh sáng yếu)
    - Fallback: Haar Cascade (khi MTCNN không khả dụng)
    
    Pipeline: RGB → MTCNN detect → bounding boxes + landmarks
    """

    def __init__(self):
        """Khởi tạo detector."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # MTCNN detector (primary)
        if MTCNN_AVAILABLE:
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.device,
                min_face_size=30,
                thresholds=[0.6, 0.7, 0.7],
                post_process=False,
            )
            print("[Detector] MTCNN initialized")
        else:
            self.mtcnn = None
            print("[Detector] MTCNN not available, using Haar Cascade only")

        # Haar Cascade (fallback)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image_np, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Phát hiện tất cả khuôn mặt trong ảnh.
        
        Args:
            image_np: ảnh RGB (numpy array)
            scale_factor: hệ số scale (chỉ dùng cho Haar fallback)
            min_neighbors: confidence threshold
            min_size: kích thước khuôn mặt tối thiểu
        
        Returns:
            list of [x, y, w, h] bounding boxes
        """
        # Try MTCNN first
        if self.mtcnn is not None:
            try:
                result = self._detect_mtcnn(image_np)
                if len(result) > 0:
                    return result
            except Exception:
                pass  # Fallback to Haar

        # Fallback: Haar Cascade
        return self._detect_haar(image_np, scale_factor, min_neighbors, min_size)

    def _detect_mtcnn(self, image_np):
        """Phát hiện bằng MTCNN (tối ưu tốc độ)."""
        # MTCNN cần PIL Image
        pil_img = Image.fromarray(image_np)
        boxes, probs = self.mtcnn.detect(pil_img)

        if boxes is None or len(boxes) == 0:
            return []

        h_img, w_img = image_np.shape[:2]
        results = []
        for box, prob in zip(boxes, probs):
            if prob < 0.5:
                continue
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            w = min(x2, w_img) - x1
            h = min(y2, h_img) - y1
            if w > 10 and h > 10:
                results.append([x1, y1, w, h])

        return results

    def _detect_haar(self, image_np, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """Phát hiện bằng Haar Cascade (fallback)."""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        if len(faces) == 0:
            return []
        return faces.tolist()

    def fast_detect_haar(self, image_np, min_size=(40, 40)):
        """
        Haar Cascade nhanh cho live camera (~5-10ms vs MTCNN ~100ms).
        Bỏ qua MTCNN hoàn toàn.
        """
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=4, minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return []
        return faces.tolist()

    @staticmethod
    def crop_face(image_np, bbox, margin=0.4):
        """
        Cắt vùng khuôn mặt với margin padding.
        
        Args:
            image_np: ảnh RGB gốc
            bbox: [x, y, w, h] bounding box
            margin: tỉ lệ margin (0.4 = 40%)
        
        Returns:
            ảnh khuôn mặt đã cắt
        """
        h, w = image_np.shape[:2]
        x, y, bw, bh = bbox
        margin_px = int(margin * max(bw, bh))

        x1 = max(0, x - margin_px)
        y1 = max(0, y - margin_px)
        x2 = min(w, x + bw + margin_px)
        y2 = min(h, y + bh + margin_px)

        return image_np[y1:y2, x1:x2]
