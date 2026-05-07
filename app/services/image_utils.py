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
    """Light preprocessing that keeps package colors but improves local contrast."""

    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_side < 900:
        scale = 900 / max_side
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def save_debug_image(image: np.ndarray, directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    cv2.imwrite(str(output_path), image)
    return output_path

