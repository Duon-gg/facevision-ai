"""
preprocessing/image_processing.py - Pipeline xử lí ảnh số (DIP)
Pipeline: Ensure RGB → CLAHE (histogram equalization) → Gaussian Blur (optional) → Resize
"""

import cv2
import numpy as np
from src.utils.constants import IMG_SIZE


# ============================================================
# Bước 1: Đảm bảo ảnh ở dạng RGB
# ============================================================

def ensure_rgb(image_np):
    """
    Chuyển đổi ảnh sang RGB nếu cần.
    - Grayscale → RGB
    - RGBA → RGB  
    - BGR → RGB (từ OpenCV)
    """
    if len(image_np.shape) == 2:
        # Grayscale → RGB
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        # RGBA → RGB
        return cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    return image_np


# ============================================================
# Bước 2: Cân bằng sáng - CLAHE trên kênh L (LAB)
# ============================================================

def apply_histogram_equalization(image_np, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Histogram equalization bằng CLAHE (Contrast Limited Adaptive HE).
    Chuyển sang LAB color space → CLAHE trên kênh L → chuyển lại RGB.
    
    Args:
        image_np: ảnh RGB (numpy array)
        clip_limit: ngưỡng cắt contrast
        tile_grid: kích thước lưới tile
    Returns:
        ảnh RGB đã cân bằng sáng
    """
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_channel = clahe.apply(l_channel)
    
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ============================================================
# Bước 3: Giảm nhiễu - Gaussian Blur (tuỳ chọn)
# ============================================================

def apply_gaussian_blur(image_np, kernel_size=3):
    """
    Lọc nhiễu Gaussian blur.
    
    Args:
        image_np: ảnh RGB
        kernel_size: kích thước kernel (phải lẻ)
    Returns:
        ảnh đã lọc nhiễu
    """
    return cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)


# ============================================================
# Bước 4: Chuẩn hoá kích thước
# ============================================================

def resize_image(image_np, target_size=IMG_SIZE):
    """
    Resize ảnh về kích thước chuẩn target_size x target_size.
    
    Args:
        image_np: ảnh RGB
        target_size: kích thước đầu ra (mặc định 128)
    Returns:
        ảnh đã resize
    """
    return cv2.resize(image_np, (target_size, target_size), interpolation=cv2.INTER_AREA)


# ============================================================
# Pipeline tổng hợp
# ============================================================

def preprocess_image(image_np, apply_blur=False, resize=True, target_size=IMG_SIZE):
    """
    Pipeline xử lí ảnh số hoàn chỉnh:
    1. Đảm bảo RGB
    2. CLAHE histogram equalization (cân bằng sáng)
    3. Gaussian blur giảm nhiễu (tuỳ chọn)
    4. Resize về kích thước chuẩn (tuỳ chọn)
    
    Args:
        image_np: ảnh đầu vào (numpy array)
        apply_blur: có áp dụng Gaussian blur không
        resize: có resize về target_size không
        target_size: kích thước đầu ra
    Returns:
        ảnh đã qua pipeline xử lí
    """
    # Bước 1: Đảm bảo RGB
    image_np = ensure_rgb(image_np)
    
    # Bước 2: CLAHE histogram equalization
    image_np = apply_histogram_equalization(image_np)
    
    # Bước 3: Gaussian blur (tuỳ chọn)
    if apply_blur:
        image_np = apply_gaussian_blur(image_np)
    
    # Bước 4: Resize
    if resize:
        image_np = resize_image(image_np, target_size)
    
    return image_np
