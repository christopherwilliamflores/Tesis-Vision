from uuid import uuid4

from fastapi import APIRouter, Depends, File, Header, Query, UploadFile

from app.api.dependencies import (
    get_product_pipeline,
    get_product_repository,
    get_suggestion_service,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import InvalidImageError
from app.repositories.products import ProductRecord, ProductRepository
from app.schemas.product import (
    ErrorResponse,
    ProductCreateRequest,
    ProductListResponse,
    ProductRecognitionResponse,
    ProductResponse,
    ProductSuggestionItemSchema,
    ProductSuggestionsResponse,
    ProductUpdateRequest,
)
from app.services.pipeline import ProductRecognitionPipeline
from app.services.suggestions import ProductSuggestionService

router = APIRouter()


def _to_response(record: ProductRecord) -> ProductResponse:
    return ProductResponse(
        id=record.id,
        nombre_producto=record.nombre_producto,
        marca=record.marca,
        tipo_producto=record.tipo_producto,
        presentacion=record.presentacion,
        contenido_neto=record.contenido_neto,
        unidad_medida=record.unidad_medida,
        categoria_sugerida=record.categoria_sugerida,
        codigo_barras=record.codigo_barras,
        precio_venta=record.precio_venta,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get("/health", tags=["health"])
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.app_env,
    }


@router.post(
    "/products/recognize",
    response_model=ProductRecognitionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Imagen inválida o demasiado grande."},
        503: {"model": ErrorResponse, "description": "Modelo YOLO/OCR no disponible."},
        500: {"model": ErrorResponse, "description": "Error interno de procesamiento."},
    },
    tags=["products"],
    summary="Reconoce datos iniciales de un producto retail peruano desde una imagen.",
)
async def recognize_product(
    image: UploadFile = File(..., description="Imagen JPG, PNG o WEBP del producto."),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
    settings: Settings = Depends(get_settings),
    pipeline: ProductRecognitionPipeline = Depends(get_product_pipeline),
) -> ProductRecognitionResponse:
    trace_id = x_trace_id or str(uuid4())
    content_type = (image.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise InvalidImageError("El archivo enviado debe ser una imagen.")

    image_bytes = await image.read()
    max_bytes = settings.max_image_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise InvalidImageError(f"La imagen supera el límite de {settings.max_image_mb} MB.")

    return pipeline.process(image_bytes=image_bytes, trace_id=trace_id, source_name=image.filename)


@router.get(
    "/productos/suggestions",
    response_model=ProductSuggestionsResponse,
    tags=["productos"],
    summary="Sugerencias de nombre de producto (≤3) sobre productos guardados con fallback al catálogo.",
)
def suggest_products(
    q: str = Query(..., min_length=0, max_length=120),
    limit: int = Query(default=3, ge=1, le=10),
    service: ProductSuggestionService = Depends(get_suggestion_service),
) -> ProductSuggestionsResponse:
    items = service.suggest(q, limit=limit)
    return ProductSuggestionsResponse(
        items=[
            ProductSuggestionItemSchema(
                nombre_producto=item.nombre_producto,
                marca=item.marca,
                tipo_producto=item.tipo_producto,
                categoria_sugerida=item.categoria_sugerida,
                source=item.source,
                product_id=item.product_id,
            )
            for item in items
        ]
    )


@router.get(
    "/productos",
    response_model=ProductListResponse,
    tags=["productos"],
    summary="Lista de productos registrados.",
)
def list_products(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: ProductRepository = Depends(get_product_repository),
) -> ProductListResponse:
    records = repository.list_all(limit=limit, offset=offset)
    return ProductListResponse(items=[_to_response(record) for record in records])


@router.get(
    "/productos/{product_id}",
    response_model=ProductResponse,
    tags=["productos"],
    responses={404: {"model": ErrorResponse}},
)
def get_product(
    product_id: int,
    repository: ProductRepository = Depends(get_product_repository),
) -> ProductResponse:
    return _to_response(repository.get(product_id))


@router.post(
    "/productos",
    response_model=ProductResponse,
    status_code=201,
    tags=["productos"],
    responses={409: {"model": ErrorResponse, "description": "Código de barras duplicado."}},
    summary="Registra un nuevo producto.",
)
def create_product(
    payload: ProductCreateRequest,
    repository: ProductRepository = Depends(get_product_repository),
) -> ProductResponse:
    return _to_response(repository.create(payload.model_dump()))


@router.put(
    "/productos/{product_id}",
    response_model=ProductResponse,
    tags=["productos"],
    responses={
        404: {"model": ErrorResponse, "description": "Producto no encontrado."},
        409: {"model": ErrorResponse, "description": "Código de barras duplicado."},
    },
    summary="Actualiza un producto existente.",
)
def update_product(
    product_id: int,
    payload: ProductUpdateRequest,
    repository: ProductRepository = Depends(get_product_repository),
) -> ProductResponse:
    return _to_response(repository.update(product_id, payload.model_dump()))
