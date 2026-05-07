import time
from dataclasses import dataclass

from app.core.config import Settings
from app.core.exceptions import ProcessingError
from app.core.logging import get_trace_logger
from app.schemas.product import (
    BoundingBox,
    DetectionMetadata,
    OcrLine,
    OcrMetadata,
    ProductRecognitionResponse,
    ProductSuggestion,
)
from app.services.detector import ProductRegionDetector, YoloRegionDetector
from app.services.image_utils import crop_image, decode_image, prepare_for_ocr, save_debug_image
from app.services.normalizer import ProductTextNormalizer
from app.services.ocr import PaddleOcrTextExtractor, TextExtractor


@dataclass
class ProductRecognitionPipeline:
    settings: Settings
    detector: ProductRegionDetector
    ocr: TextExtractor
    normalizer: ProductTextNormalizer

    def process(
        self,
        image_bytes: bytes,
        trace_id: str,
        source_name: str | None = None,
    ) -> ProductRecognitionResponse:
        logger = get_trace_logger(__name__, trace_id)
        start = time.perf_counter()
        logger.info("inicio procesamiento producto")

        image = decode_image(image_bytes)
        logger.info("imagen decodificada")

        detection = self.detector.detect(image)
        logger.info(
            "deteccion yolo completada bbox=%s confidence=%s fallback=%s",
            detection.bbox,
            detection.confidence,
            detection.used_full_image_fallback,
        )

        cropped = crop_image(image, detection.bbox)
        ocr_image = prepare_for_ocr(cropped)
        if self.settings.persist_debug_images:
            save_debug_image(ocr_image, self.settings.debug_image_dir, f"{trace_id}_roi.jpg")

        ocr_result = self.ocr.extract(ocr_image)
        logger.info("ocr completado lineas=%s", len(ocr_result.lines))

        normalized = self.normalizer.normalize(ocr_result.text, source_name=source_name)
        warnings = list(normalized.warnings)
        if detection.used_full_image_fallback:
            warnings.append("YOLO no detectó una región; se procesó la imagen completa como respaldo técnico.")
        if not ocr_result.lines:
            warnings.append("OCR sin líneas detectadas; el producto debe revisarse manualmente.")

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        if elapsed_ms < 0:  # pragma: no cover - defensive
            raise ProcessingError("Tiempo de procesamiento inválido.")

        return ProductRecognitionResponse(
            trace_id=trace_id,
            producto=ProductSuggestion(
                nombre_producto=normalized.nombre_producto,
                marca=normalized.marca,
                tipo_producto=normalized.tipo_producto,
                presentacion=normalized.presentacion,
                contenido_neto=normalized.contenido_neto,
                unidad_medida=normalized.unidad_medida,
                categoria_sugerida=normalized.categoria_sugerida,
            ),
            deteccion=DetectionMetadata(
                model=detection.model,
                bbox=BoundingBox(
                    x_min=detection.bbox[0],
                    y_min=detection.bbox[1],
                    x_max=detection.bbox[2],
                    y_max=detection.bbox[3],
                ),
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name,
                used_full_image_fallback=detection.used_full_image_fallback,
            ),
            ocr=OcrMetadata(
                engine=ocr_result.engine,
                text=ocr_result.text,
                average_confidence=ocr_result.average_confidence,
                lines=[
                    OcrLine(text=line.text, confidence=line.confidence)
                    for line in ocr_result.lines
                ],
            ),
            warnings=warnings,
            processing_ms=elapsed_ms,
        )


def build_pipeline(settings: Settings) -> ProductRecognitionPipeline:
    if settings.ocr_engine.lower() != "paddle":
        raise ProcessingError(f"OCR_ENGINE no soportado: {settings.ocr_engine}")

    return ProductRecognitionPipeline(
        settings=settings,
        detector=YoloRegionDetector(settings),
        ocr=PaddleOcrTextExtractor(settings),
        normalizer=ProductTextNormalizer(),
    )
