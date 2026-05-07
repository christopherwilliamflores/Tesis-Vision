from fastapi.testclient import TestClient

from app.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _create_payload(**overrides) -> dict:
    payload = {
        "nombre_producto": "Inca Kola Botella 500 ml",
        "marca": "Inca Kola",
        "tipo_producto": "Gaseosa",
        "presentacion": "botella 500 ml",
        "contenido_neto": "500 ml",
        "unidad_medida": "ml",
        "categoria_sugerida": "bebidas",
        "codigo_barras": "7751234500011",
        "precio_venta": 4.5,
    }
    payload.update(overrides)
    return payload


def test_create_and_get_product() -> None:
    client = _client()
    response = client.post("/api/v1/productos", json=_create_payload())
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"]
    assert body["precio_venta"] == 4.5

    fetched = client.get(f"/api/v1/productos/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["nombre_producto"] == "Inca Kola Botella 500 ml"


def test_create_rejects_missing_required_fields() -> None:
    client = _client()
    response = client.post(
        "/api/v1/productos",
        json={"nombre_producto": "  ", "categoria_sugerida": "bebidas", "precio_venta": 1.0},
    )
    assert response.status_code == 422


def test_create_rejects_negative_price() -> None:
    client = _client()
    response = client.post("/api/v1/productos", json=_create_payload(precio_venta=-1))
    assert response.status_code == 422


def test_duplicate_barcode_returns_409() -> None:
    client = _client()
    first = client.post("/api/v1/productos", json=_create_payload())
    assert first.status_code == 201
    second = client.post(
        "/api/v1/productos",
        json=_create_payload(nombre_producto="Otro nombre"),
    )
    assert second.status_code == 409
    assert second.json()["error_code"] == "DUPLICATE_BARCODE"


def test_update_product_modifies_fields() -> None:
    client = _client()
    created = client.post("/api/v1/productos", json=_create_payload()).json()
    response = client.put(
        f"/api/v1/productos/{created['id']}",
        json=_create_payload(precio_venta=5.9, nombre_producto="Inca Kola 500 ml"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["precio_venta"] == 5.9
    assert body["nombre_producto"] == "Inca Kola 500 ml"


def test_update_missing_product_returns_404() -> None:
    client = _client()
    response = client.put("/api/v1/productos/9999", json=_create_payload())
    assert response.status_code == 404
    assert response.json()["error_code"] == "PRODUCT_NOT_FOUND"


def test_suggestions_below_three_chars_returns_empty() -> None:
    client = _client()
    response = client.get("/api/v1/productos/suggestions", params={"q": "in"})
    assert response.status_code == 200
    assert response.json()["items"] == []


def test_suggestions_returns_db_match_first() -> None:
    client = _client()
    client.post("/api/v1/productos", json=_create_payload())
    response = client.get("/api/v1/productos/suggestions", params={"q": "inca"})
    assert response.status_code == 200
    items = response.json()["items"]
    assert items, items
    assert items[0]["source"] == "db"
    assert "Inca Kola" in items[0]["nombre_producto"]


def test_suggestions_falls_back_to_catalog() -> None:
    client = _client()
    response = client.get("/api/v1/productos/suggestions", params={"q": "gas"})
    assert response.status_code == 200
    items = response.json()["items"]
    assert items, items
    assert all(item["source"] == "catalog" for item in items)
    assert any(item["nombre_producto"].lower().startswith("gaseosa") for item in items)


def test_suggestions_capped_to_three() -> None:
    client = _client()
    response = client.get(
        "/api/v1/productos/suggestions", params={"q": "ola", "limit": 10}
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) <= 10  # respects limit, but UI uses 3
    response_three = client.get(
        "/api/v1/productos/suggestions", params={"q": "ola", "limit": 3}
    )
    assert len(response_three.json()["items"]) <= 3


def test_list_products_returns_recent_first() -> None:
    client = _client()
    first = client.post("/api/v1/productos", json=_create_payload()).json()
    second = client.post(
        "/api/v1/productos",
        json=_create_payload(codigo_barras="7751234500022", nombre_producto="Field Galleta"),
    ).json()
    response = client.get("/api/v1/productos")
    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["items"]]
    assert ids[0] == second["id"]
    assert first["id"] in ids
