from pathlib import Path

import cv2
import numpy as np

from app.core.exceptions import InvalidImageError


def decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise InvalidImageError("La imagen enviada está vacía.")

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("No se pudo decodificar la imagen. Usa JPG, PNG o WEBP válido.")
    return image


def crop_image(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, min(width - 1, x_min))
    x_max = max(1, min(width, x_max))
    y_min = max(0, min(height - 1, y_min))
    y_max = max(1, min(height, y_max))
    if x_max <= x_min or y_max <= y_min:
        raise InvalidImageError("La región detectada no es válida para recorte.")
    return image[y_min:y_max, x_min:x_max]


def prepare_for_ocr(image: np.ndarray) -> np.ndarray:
    """Preprocess package crops for OCR while preserving text color cues."""

    image = trim_plain_background(image)
    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_side < 1400:
        scale = 1400 / max_side
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)
    return sharpened


def trim_plain_background(image: np.ndarray) -> np.ndarray:
    """Remove wide plain margins so OCR spends resolution on the package text."""

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    foreground = (saturation > 18) | (value < 242) | (gray < 238)
    foreground = foreground.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    crop_area = w * h
    image_area = width * height
    if crop_area < image_area * 0.03 or crop_area > image_area * 0.96:
        return image

    margin = max(8, int(max(width, height) * 0.04))
    x_min = max(0, x - margin)
    y_min = max(0, y - margin)
    x_max = min(width, x + w + margin)
    y_max = min(height, y + h + margin)
    return image[y_min:y_max, x_min:x_max]


def save_debug_image(image: np.ndarray, directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    cv2.imwrite(str(output_path), image)
    return output_path

