# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

MVP for the thesis *"Sistema inteligente para optimizar el proceso de gestión e inventario de productos usando redes neuronales en empresas PYMES del sector retail"*. The system accepts a product image, detects the package/label region with YOLO, runs PaddleOCR on the crop, normalizes the text against a Peruvian retail catalog, and returns a structured JSON suggestion for product registration.

Domain language is Spanish (Peruvian retail). Keep response fields, warnings, and user-facing strings in Spanish to match existing code.

## Common commands

```bash
# Local dev (Python 3.10–3.12)
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload          # http://localhost:8000

# Tests
pytest                                  # config in pyproject.toml (testpaths=tests, -q)
pytest tests/test_api.py::test_recognize_product_returns_structured_json

# Docker
docker compose up --build               # mounts ./runtime and ./models

# Train YOLO on demo/ images (full-image bbox, class product_label)
python scripts/prepare_demo_yolo_dataset.py --source demo --output data/demo_yolo
python scripts/train_yolo_demo.py --data data/demo_yolo/dataset.yaml --epochs 15 --imgsz 416 --device cpu
# Then point the API at the trained weights:
YOLO_MODEL_PATH=models/product_label_demo.pt uvicorn app.main:app --reload
```

There is no linter or formatter configured.

## Architecture

`app/main.py:create_app` wires FastAPI: API router under `settings.api_v1_prefix` (default `/api/v1`), `/static` mount, `/` serves `app/web/index.html`, and two exception handlers (`AppError` → structured `ErrorResponse`, generic `Exception` → 500 `UNHANDLED_ERROR`). Both handlers echo the `X-Trace-ID` request header into the response.

The single non-trivial endpoint is `POST /api/v1/products/recognize` (`app/api/routes.py`). It validates content-type and `MAX_IMAGE_MB`, generates a trace id if `X-Trace-ID` is missing, then delegates to the pipeline.

### The pipeline (`app/services/pipeline.py`)

`ProductRecognitionPipeline.process` is the spine. Order matters — downstream steps depend on prior outputs:

1. `decode_image` (OpenCV) → raises `InvalidImageError` on bad bytes.
2. `detector.detect` → `RegionDetection`. `YoloRegionDetector` lazy-loads `ultralytics.YOLO(YOLO_MODEL_PATH)`. If no boxes are returned and `ALLOW_FULL_IMAGE_FALLBACK=true`, returns the full image bbox with `used_full_image_fallback=True`; otherwise raises `ProcessingError`. Best box is picked by `area * confidence`, not confidence alone.
3. `crop_image` + `prepare_for_ocr` (upscale to ≥900px on the long side, CLAHE on the L channel of LAB).
4. `ocr.extract` → `PaddleOcrTextExtractor` lazy-loads `PaddleOCR(lang=PADDLE_OCR_LANG)`. The OCR wrapper handles **both** PaddleOCR v2 (`use_angle_cls`, `client.ocr(...)` returning nested lists) and v3 (`use_textline_orientation`, `client.predict(...)` returning dicts with `rec_texts`/`rec_scores`). Don't simplify — both branches are needed in practice.
5. `normalizer.normalize(text, source_name)` → `NormalizedProduct`.
6. Pipeline appends pipeline-level warnings (fallback used, no OCR lines) on top of normalizer warnings.

The pipeline is constructed once via `build_pipeline(settings)` and cached by `app/api/dependencies.py:_cached_pipeline` (`lru_cache`). Override it in tests with `app.dependency_overrides[get_product_pipeline] = lambda: fake_pipeline` (see `tests/test_api.py`). YOLO and PaddleOCR are **never imported at module load** — only on first `.model` / `.client` access — so unit tests can build the pipeline without those heavy deps as long as fakes are injected.

### Normalizer (`app/services/normalizer.py` + `app/domain/catalog.py`)

The normalizer is rule-based, not ML, and is the place most thesis-relevant changes will land. It folds text to ASCII-lowercase, then matches against catalogs in `app/domain/catalog.py`:

- `BRAND_ALIASES` — canonical brand + aliases + optional `category_hint`.
- `PRODUCT_TYPE_ALIASES`, `BRAND_DEFAULT_PRODUCT_TYPES`, `PRODUCT_TYPE_CATEGORIES` — type detection and type→category fallback.
- `CATEGORY_KEYWORDS`, `VARIANT_ALIASES`, `UNIT_ALIASES`, `NOISE_PHRASES`, `PRODUCT_STOPWORDS`.

When extending detection, prefer adding entries to `app/domain/catalog.py` over adding regex branches in `normalizer.py`. The `source_name` argument (the uploaded filename) is folded into the matching context — useful signal, but strip extensions/numbers via `_clean_source_name`.

Quantity/unit detection prefers mass/volume units (`g`, `kg`, `ml`, `L`) over count units when multiple matches exist (`_detect_content` scoring). Each unmet field appends a Spanish warning rather than failing; the API is best-effort by design.

### Errors and tracing

All expected failures inherit `AppError` (`app/core/exceptions.py`) with a `status_code` and `error_code` (`INVALID_IMAGE` 400, `MODEL_UNAVAILABLE` 503, `PROCESSING_ERROR` 500). Raise these — don't return error JSON manually; the handler in `main.py` formats `ErrorResponse` and logs with `trace_id`. Logs use `get_trace_logger(__name__, trace_id)` so the `trace_id=...` field flows through the formatter configured in `app/core/logging.py`.

### Configuration

`Settings` (`app/core/config.py`, `pydantic-settings`) loads from env / `.env`. `get_settings()` is `lru_cache`d, so changing env vars at runtime requires restarting the process. New config goes here, not as constants scattered through services. Key vars: `YOLO_MODEL_PATH`, `YOLO_CONFIDENCE_THRESHOLD`, `YOLO_DEVICE`, `OCR_ENGINE` (only `paddle` is implemented — `build_pipeline` raises `ProcessingError` otherwise), `PADDLE_OCR_LANG`, `ALLOW_FULL_IMAGE_FALLBACK`, `PERSIST_DEBUG_IMAGES`, `MAX_IMAGE_MB`.

## Testing notes

Tests in `tests/` mock the detector and OCR via the `FakeDetector` / `FakeOcr` pattern and inject a `ProductRecognitionPipeline` through `dependency_overrides`. This keeps the suite fast and avoids downloading YOLO / Paddle weights. Follow this pattern for new pipeline-level tests; don't add tests that require real model downloads.
