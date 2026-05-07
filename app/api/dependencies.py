from functools import lru_cache

from app.core.config import get_settings
from app.repositories.products import ProductRepository
from app.services.pipeline import ProductRecognitionPipeline, build_pipeline
from app.services.suggestions import ProductSuggestionService, build_suggestion_service


@lru_cache
def _cached_pipeline() -> ProductRecognitionPipeline:
    return build_pipeline(get_settings())


@lru_cache
def _cached_repository() -> ProductRepository:
    return ProductRepository(get_settings())


@lru_cache
def _cached_suggestion_service() -> ProductSuggestionService:
    return build_suggestion_service(_cached_repository())


def get_product_pipeline() -> ProductRecognitionPipeline:
    return _cached_pipeline()


def get_product_repository() -> ProductRepository:
    return _cached_repository()


def get_suggestion_service() -> ProductSuggestionService:
    return _cached_suggestion_service()
