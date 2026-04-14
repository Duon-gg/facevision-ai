"""
inference/analyzer.py - Pipeline suy luận: Detection → Preprocess → CNN Predict
Hỗ trợ phân tích ảnh tĩnh và frame video real-time
Test-Time Augmentation (TTA) cho dự đoán chính xác hơn
"""

import os
import json
import cv2
import numpy as np
from PIL import Image

import torch
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from torchvision import transforms

from src.models.face_attribute_model import FaceAttributeModel
from src.detection.face_detector import FaceDetector
from src.preprocessing.image_processing import preprocess_image
from src.utils.constants import (
    GENDER_LABELS, RACE_LABELS, AGE_GROUPS,
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BBOX_COLORS, NUM_RACE_CLASSES, age_to_group
)


# Inference transform (no augmentation)
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


class FaceAnalyzer:
    """
    Pipeline suy luận hoàn chỉnh:
    1. Phát hiện khuôn mặt (MTCNN + Haar Cascade fallback)
    2. Cắt và xử lí ảnh (DIP pipeline)
    3. Dự đoán đa nhiệm (Age, Gender, Race) với xác suất
    4. Test-Time Augmentation (TTA) cho độ chính xác cao hơn
    
    Args:
        model_path: đường dẫn file model .pth
        device: 'cuda' hoặc 'cpu'
        age_mode: 'regression' hoặc 'age_group'
    """

    def __init__(self, model_path='checkpoints/best_model.pth', device=None, age_mode='regression'):
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.age_mode = age_mode

        # Load model
        self.model = FaceAttributeModel(pretrained=False, age_mode=age_mode)
        self.model = self.model.to(self.device)
        self.model_loaded = False

        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                state_dict = checkpoint['model_state_dict']
                
                # Remap keys: Colab dùng 'se.exc' → local dùng 'se.excitation'
                remapped = {}
                for k, v in state_dict.items():
                    new_key = k.replace('se.exc.', 'se.excitation.')
                    remapped[new_key] = v
                
                self.model.load_state_dict(remapped)
                self.model.eval()
                self.model_loaded = True
                version = checkpoint.get('model_version', 'unknown')
                print(f"[Model] Loaded from {model_path} (version: {version})")
            except Exception as e:
                print(f"[Warning] Failed to load model: {e}")
                print("[Warning] Running in DEMO mode (random predictions)")
        else:
            print(f"[Warning] Model file not found: {model_path}")
            print("[Warning] Running in DEMO mode (random predictions)")

        # Face detector
        self.detector = FaceDetector()

        # Race threshold calibration
        self.race_thresholds = None
        thresholds_path = os.path.join(os.path.dirname(model_path), '..', 'config', 'race_thresholds.json')
        if os.path.exists(thresholds_path):
            try:
                with open(thresholds_path, 'r', encoding='utf-8') as f:
                    th_config = json.load(f)
                self.race_thresholds = np.array(th_config['threshold_array'], dtype=np.float32)
                print(f"[Calibration] Loaded race thresholds: {th_config['thresholds']}")
            except Exception as e:
                print(f"[Calibration] Failed to load thresholds: {e}")

    def _prepare_tensor(self, face_np, apply_blur=False):
        """Chuyển numpy array → tensor cho model (full preprocessing)."""
        face_processed = preprocess_image(
            face_np.copy(), apply_blur=apply_blur, resize=True
        )
        face_pil = Image.fromarray(face_processed)
        return INFERENCE_TRANSFORM(face_pil).unsqueeze(0).to(self.device)

    def _prepare_tensor_fast(self, face_np):
        """Chuyển numpy → tensor NHANH (skip CLAHE, chỉ resize+normalize)."""
        face_pil = Image.fromarray(face_np)
        return INFERENCE_TRANSFORM(face_pil).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _predict_single(self, face_tensor):
        """Predict trên 1 tensor, trả về raw outputs."""
        with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
            age_pred, gender_pred, race_pred = self.model(face_tensor)
        return age_pred, gender_pred, race_pred

    @torch.no_grad()
    def predict_face(self, face_np, apply_blur=False, use_tta=False):
        """
        Dự đoán thuộc tính cho một khuôn mặt đã cắt.
        
        Args:
            face_np: ảnh khuôn mặt RGB (numpy array)
            apply_blur: áp dụng Gaussian blur
            use_tta: sử dụng Test-Time Augmentation
        
        Returns:
            dict với age, gender, race và xác suất chi tiết
        """
        if not self.model_loaded:
            return {
                'age': np.random.randint(10, 60),
                'age_group': 'Young Adult',
                'gender': 'Male',
                'gender_idx': 0,
                'gender_prob': [0.5, 0.5],
                'race': 'Asian',
                'race_idx': 2,
                'race_prob': [0.25] * 4,
            }

        if use_tta:
            return self._predict_with_tta(face_np, apply_blur)
        
        # Standard prediction
        face_tensor = self._prepare_tensor(face_np, apply_blur)
        age_pred, gender_pred, race_pred = self._predict_single(face_tensor)
        return self._parse_predictions(age_pred, gender_pred, race_pred)

    def _predict_with_tta(self, face_np, apply_blur=False):
        """
        Test-Time Augmentation: chạy 5 augmentations, lấy trung bình.
        Augmentations: original, horizontal flip, brightness+10%, brightness-10%, slight rotate
        """
        augmented_images = [face_np]  # original

        # 1. Horizontal flip
        augmented_images.append(np.fliplr(face_np).copy())

        # 2. Brightness + 10%
        bright_up = np.clip(face_np.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
        augmented_images.append(bright_up)

        # 3. Brightness - 10%
        bright_down = np.clip(face_np.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
        augmented_images.append(bright_down)

        # 4. Slight rotation (5 degrees)
        h, w = face_np.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 5, 1.0)
        rotated = cv2.warpAffine(face_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented_images.append(rotated)

        # Collect predictions
        all_age = []
        all_gender_prob = []
        all_race_prob = []

        for aug_img in augmented_images:
            face_tensor = self._prepare_tensor(aug_img, apply_blur)
            age_pred, gender_pred, race_pred = self._predict_single(face_tensor)

            # Age
            if self.age_mode == 'regression':
                all_age.append(age_pred.item())
            else:
                all_age.append(torch.softmax(age_pred, dim=1).cpu().numpy()[0])

            # Gender & Race probs
            all_gender_prob.append(torch.softmax(gender_pred, dim=1).cpu().numpy()[0])
            all_race_prob.append(torch.softmax(race_pred, dim=1).cpu().numpy()[0])

        # Average predictions
        if self.age_mode == 'regression':
            age = max(0, min(120, int(np.mean(all_age))))
            age_group = AGE_GROUPS[age_to_group(age)]
        else:
            avg_age_probs = np.mean(all_age, axis=0)
            age_group_idx = int(avg_age_probs.argmax())
            age_group = AGE_GROUPS[age_group_idx]
            age_centers = [6, 16, 27, 45, 65]
            age = age_centers[age_group_idx]

        avg_gender = np.mean(all_gender_prob, axis=0)
        gender_idx = int(avg_gender.argmax())
        gender_label = GENDER_LABELS[gender_idx]

        avg_race = np.mean(all_race_prob, axis=0)
        # Apply threshold calibration
        if self.race_thresholds is not None:
            avg_race = avg_race * self.race_thresholds
        race_idx = int(avg_race.argmax())
        race_label = RACE_LABELS[race_idx]

        return {
            'age': age,
            'age_group': age_group,
            'gender': gender_label,
            'gender_idx': gender_idx,
            'gender_prob': avg_gender.tolist(),
            'race': race_label,
            'race_idx': race_idx,
            'race_prob': avg_race.tolist(),
        }

    def _parse_predictions(self, age_pred, gender_pred, race_pred):
        """Parse raw model outputs thành dict kết quả."""
        # Age
        if self.age_mode == 'regression':
            age = max(0, min(120, int(age_pred.item())))
            age_group = AGE_GROUPS[age_to_group(age)]
        else:
            age_probs = torch.softmax(age_pred, dim=1).cpu().numpy()[0]
            age_group_idx = int(age_probs.argmax())
            age_group = AGE_GROUPS[age_group_idx]
            age_centers = [6, 16, 27, 45, 65]
            age = age_centers[age_group_idx]

        # Gender
        gender_probs = torch.softmax(gender_pred, dim=1).cpu().numpy()[0]
        gender_idx = int(gender_probs.argmax())
        gender_label = GENDER_LABELS[gender_idx]

        # Race
        race_probs = torch.softmax(race_pred, dim=1).cpu().numpy()[0]
        # Apply threshold calibration
        if self.race_thresholds is not None:
            race_probs = race_probs * self.race_thresholds
        race_idx = int(race_probs.argmax())
        race_label = RACE_LABELS[race_idx]

        return {
            'age': age,
            'age_group': age_group,
            'gender': gender_label,
            'gender_idx': gender_idx,
            'gender_prob': gender_probs.tolist(),
            'race': race_label,
            'race_idx': race_idx,
            'race_prob': race_probs.tolist(),
        }

    @torch.no_grad()
    def predict_faces_batch(self, face_crops, apply_blur=False):
        """
        Batch prediction: xử lý nhiều khuôn mặt trong 1 forward pass GPU.
        Nhanh hơn nhiều so với predict từng face riêng lẻ.
        """
        if not self.model_loaded or len(face_crops) == 0:
            return [self.predict_face(f, apply_blur=apply_blur) for f in face_crops]

        tensors = [self._prepare_tensor(f, apply_blur) for f in face_crops]
        batch = torch.cat(tensors, dim=0)

        with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
            age_pred, gender_pred, race_pred = self.model(batch)

        results = []
        for i in range(len(face_crops)):
            results.append(self._parse_predictions(
                age_pred[i:i+1], gender_pred[i:i+1], race_pred[i:i+1]
            ))
        return results

    @torch.no_grad()
    def predict_faces_batch_fast(self, face_crops):
        """
        Batch prediction NHANH cho live camera.
        Skip CLAHE preprocessing → tiết kiệm ~10ms/face.
        """
        if not self.model_loaded or len(face_crops) == 0:
            return [self.predict_face(f) for f in face_crops]

        tensors = [self._prepare_tensor_fast(f) for f in face_crops]
        batch = torch.cat(tensors, dim=0)

        with autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
            age_pred, gender_pred, race_pred = self.model(batch)

        results = []
        for i in range(len(face_crops)):
            results.append(self._parse_predictions(
                age_pred[i:i+1], gender_pred[i:i+1], race_pred[i:i+1]
            ))
        return results

    def analyze_image(self, image_input, min_neighbors=5, apply_blur=False, use_tta=True):
        """
        Pipeline phân tích hoàn chỉnh: detect → crop → predict.
        
        Args:
            image_input: str (file path), np.ndarray (RGB), hoặc PIL.Image
            min_neighbors: confidence threshold cho face detection
            apply_blur: áp dụng Gaussian blur
            use_tta: sử dụng Test-Time Augmentation (chậm hơn nhưng chính xác hơn)
        
        Returns:
            results: list of dicts (mỗi dict là một khuôn mặt)
            image_np: ảnh gốc dạng numpy RGB
        """
        # Load image
        if isinstance(image_input, str):
            image_np = cv2.imread(image_input)
            if image_np is None:
                return [], None
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image_np = np.array(image_input)
        else:
            image_np = image_input.copy()

        # Ensure RGB
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        # Detect faces
        bboxes = self.detector.detect_faces(image_np, min_neighbors=min_neighbors)

        # Nếu không phát hiện khuôn mặt → dùng toàn bộ ảnh
        if len(bboxes) == 0:
            h, w = image_np.shape[:2]
            bboxes = [[0, 0, w, h]]

        results = []
        for bbox in bboxes:
            face_crop = self.detector.crop_face(image_np, bbox)
            if face_crop.size == 0:
                continue
            prediction = self.predict_face(
                face_crop, apply_blur=apply_blur, use_tta=use_tta
            )
            prediction['bbox'] = bbox
            results.append(prediction)

        return results, image_np

    def fast_analyze(self, image_np, bboxes=None, min_neighbors=5, apply_blur=False):
        """
        Pipeline SIÊU NHANH cho live camera:
        - Dùng Haar Cascade thay MTCNN (~5ms vs ~100ms)
        - Skip CLAHE preprocessing (~10ms/face)
        - Batch predict all faces trong 1 forward pass
        - Không copy ảnh, không TTA
        
        Args:
            image_np: ảnh RGB (numpy array), KHÔNG copy
            bboxes: bounding boxes đã detect (None = detect mới)
            min_neighbors: confidence threshold
            apply_blur: áp dụng Gaussian blur (ignored in fast mode)
        
        Returns:
            results: list of dicts
            bboxes: bounding boxes (để reuse frame sau)
        """
        # Detect bằng Haar Cascade (siêu nhanh)
        if bboxes is None:
            bboxes = self.detector.fast_detect_haar(image_np)
            if len(bboxes) == 0:
                return [], []

        # Crop tất cả faces
        face_crops = []
        valid_bboxes = []
        for bbox in bboxes:
            crop = self.detector.crop_face(image_np, bbox)
            if crop.size > 0:
                face_crops.append(crop)
                valid_bboxes.append(bbox)

        if not face_crops:
            return [], []

        # Batch predict NHANH (skip CLAHE)
        predictions = self.predict_faces_batch_fast(face_crops)
        for pred, bbox in zip(predictions, valid_bboxes):
            pred['bbox'] = bbox

        return predictions, valid_bboxes

    def draw_results(self, image_np, results):
        """
        Vẽ bounding box và nhãn lên ảnh.
        
        Args:
            image_np: ảnh RGB gốc
            results: list kết quả từ analyze_image
        
        Returns:
            ảnh đã vẽ annotation
        """
        img_draw = image_np.copy()

        for r in results:
            x, y, w, h = r['bbox']
            gender = r.get('gender', 'Male')

            # Màu theo giới tính
            color = BBOX_COLORS.get(gender, (0, 200, 0))

            # Vẽ bounding box
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 2)

            # Label với confidence
            gender_conf = max(r.get('gender_prob', [0.5, 0.5])) * 100
            race_conf = max(r.get('race_prob', [0.25] * 4)) * 100
            label = f"{r['gender']}, {r['age']}, {r['race']} ({race_conf:.0f}%)"
            
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Background cho text
            cv2.rectangle(
                img_draw,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                color, -1
            )
            cv2.putText(
                img_draw, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        return img_draw


if __name__ == "__main__":
    # Quick test
    analyzer = FaceAnalyzer()
    test_img = "data/UTKFace/10_0_0_20161220222308131.jpg.chip.jpg"
    if os.path.exists(test_img):
        results, img = analyzer.analyze_image(test_img, use_tta=True)
        for r in results:
            print(f"Age: {r['age']}, Gender: {r['gender']}, Race: {r['race']}")
            print(f"  Gender probs: {r['gender_prob']}")
            print(f"  Race probs: {r['race_prob']}")
    else:
        print(f"Test image not found: {test_img}")
