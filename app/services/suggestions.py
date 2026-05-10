import unicodedata
from dataclasses import dataclass
import re

from app.domain.catalog import (
    BRAND_ALIASES,
    BRAND_DEFAULT_PRODUCT_TYPES,
    PRODUCT_TYPE_ALIASES,
    PRODUCT_TYPE_CATEGORIES,
)
from app.repositories.products import ProductRepository, ProductRecord
from app.services.normalizer import ProductTextNormalizer


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
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char)).strip()
    return re.sub(r"\s+", " ", ascii_text)


def _append_unique(names: list[str], name: str | None) -> None:
    if not name:
        return
    clean_name = " ".join(name.split())
    if clean_name and _fold(clean_name) not in {_fold(item) for item in names}:
        names.append(clean_name)


class ProductSuggestionService:
    def __init__(self, repository: ProductRepository) -> None:
        self.repository = repository
        self.normalizer = ProductTextNormalizer()

    def suggest(
        self,
        query: str,
        limit: int = 3,
        context_text: str | None = None,
        source_name: str | None = None,
    ) -> list[ProductSuggestionItem]:
        query = (query or "").strip()
        if len(query) < 3:
            return []

        suggestions: list[ProductSuggestionItem] = []
        seen: set[str] = set()

        for item in self._context_candidates(query, context_text, source_name):
            key = _fold(item.nombre_producto)
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(item)
            if len(suggestions) >= limit:
                return suggestions

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

    def _context_candidates(
        self,
        query: str,
        context_text: str | None,
        source_name: str | None,
    ) -> list[ProductSuggestionItem]:
        if not context_text:
            return []

        normalized = self.normalizer.normalize(context_text, source_name=source_name)
        base_name = normalized.nombre_producto
        if not base_name:
            return []

        names = self._context_names(base_name, normalized, context_text)
        if normalized.contenido_neto and _fold(normalized.contenido_neto) not in _fold(base_name):
            _append_unique(names, f"{base_name} {normalized.contenido_neto}")

        items: list[ProductSuggestionItem] = []
        for name in names[:3]:
            items.append(
                ProductSuggestionItem(
                    nombre_producto=name,
                    marca=normalized.marca,
                    tipo_producto=normalized.tipo_producto,
                    categoria_sugerida=normalized.categoria_sugerida,
                    source="ocr",
                )
            )
        return items

    def _context_names(self, base_name: str, normalized, context_text: str) -> list[str]:
        names: list[str] = []
        _append_unique(names, base_name)

        product_type = normalized.tipo_producto
        brand = normalized.marca
        if product_type and brand:
            folded_context = _fold(context_text)
            phrase_options = [
                ("rosas y magnolias", "Rosas y Magnolias"),
                ("cuidadoy suavidad", "Cuidado y Suavidad"),
                ("cuidado y suavidad", "Cuidado y Suavidad"),
                ("cuidado total", "Cuidado Total"),
                ("suavidad y fragancia prolongada", "Suavidad y Fragancia Prolongada"),
                ("suavidad fragancia prolongada", "Suavidad y Fragancia Prolongada"),
                ("protege el color y las fibras", "Protege Color y Fibras"),
                ("protege color fibras", "Protege Color y Fibras"),
                ("elcolory lasfibras", "Protege Color y Fibras"),
                ("lavanda", "Lavanda"),
                ("lawanda", "Lavanda"),
            ]
            for needle, label in phrase_options:
                if needle in folded_context:
                    _append_unique(names, f"{product_type} {brand} {label}")
            if "cuidado" in folded_context and "total" in folded_context:
                _append_unique(names, f"{product_type} {brand} Cuidado Total")
            if "protege" in folded_context and "color" in folded_context and "fibras" in folded_context:
                _append_unique(names, f"{product_type} {brand} Protege Color y Fibras")
            if "suavidad" in folded_context and "fragancia" in folded_context and "prolongada" in folded_context:
                _append_unique(names, f"{product_type} {brand} Suavidad y Fragancia Prolongada")

            if normalized.contenido_neto and _fold(normalized.contenido_neto) not in _fold(base_name):
                _append_unique(names, f"{base_name} {normalized.contenido_neto}")

            if normalized.presentacion:
                _append_unique(names, f"{product_type} {brand} {normalized.presentacion}")

            _append_unique(names, f"{product_type} {brand}")

        return names

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
