import unicodedata
from dataclasses import dataclass

from app.domain.catalog import (
    BRAND_ALIASES,
    BRAND_DEFAULT_PRODUCT_TYPES,
    PRODUCT_TYPE_ALIASES,
    PRODUCT_TYPE_CATEGORIES,
)
from app.repositories.products import ProductRepository, ProductRecord


@dataclass(frozen=True)
class ProductSuggestionItem:
    nombre_producto: str
    marca: str | None = None
    tipo_producto: str | None = None
    categoria_sugerida: str | None = None
    source: str = "catalog"
    product_id: int | None = None


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char)).strip()


class ProductSuggestionService:
    def __init__(self, repository: ProductRepository) -> None:
        self.repository = repository

    def suggest(self, query: str, limit: int = 3) -> list[ProductSuggestionItem]:
        query = (query or "").strip()
        if len(query) < 3:
            return []

        suggestions: list[ProductSuggestionItem] = []
        seen: set[str] = set()

        for record in self.repository.search_by_name(query, limit=limit):
            key = _fold(record.nombre_producto)
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(
                ProductSuggestionItem(
                    nombre_producto=record.nombre_producto,
                    marca=record.marca,
                    tipo_producto=record.tipo_producto,
                    categoria_sugerida=record.categoria_sugerida,
                    source="db",
                    product_id=record.id,
                )
            )
            if len(suggestions) >= limit:
                return suggestions

        for item in self._catalog_candidates(query):
            key = _fold(item.nombre_producto)
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(item)
            if len(suggestions) >= limit:
                break

        return suggestions

    def _catalog_candidates(self, query: str) -> list[ProductSuggestionItem]:
        folded_query = _fold(query)
        items: list[tuple[int, ProductSuggestionItem]] = []

        for product_type, aliases in PRODUCT_TYPE_ALIASES.items():
            for alias in (product_type, *aliases):
                folded_alias = _fold(alias)
                if folded_query in folded_alias:
                    score = 0 if folded_alias.startswith(folded_query) else 1
                    items.append(
                        (
                            score,
                            ProductSuggestionItem(
                                nombre_producto=product_type,
                                tipo_producto=product_type,
                                categoria_sugerida=PRODUCT_TYPE_CATEGORIES.get(product_type),
                                source="catalog",
                            ),
                        )
                    )
                    break

        for brand in BRAND_ALIASES:
            for alias in (brand.canonical, *brand.aliases):
                folded_alias = _fold(alias)
                if folded_query in folded_alias:
                    score = 0 if folded_alias.startswith(folded_query) else 1
                    items.append(
                        (
                            score,
                            ProductSuggestionItem(
                                nombre_producto=brand.canonical,
                                marca=brand.canonical,
                                tipo_producto=BRAND_DEFAULT_PRODUCT_TYPES.get(brand.canonical),
                                categoria_sugerida=brand.category_hint,
                                source="catalog",
                            ),
                        )
                    )
                    break

        items.sort(key=lambda pair: (pair[0], len(pair[1].nombre_producto)))
        return [item for _, item in items]


def build_suggestion_service(repository: ProductRepository) -> ProductSuggestionService:
    return ProductSuggestionService(repository)
