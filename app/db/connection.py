import sqlite3
from pathlib import Path
from threading import Lock

from app.core.config import Settings, get_settings

_lock = Lock()
_initialized: set[str] = set()


SCHEMA = """
CREATE TABLE IF NOT EXISTS productos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre_producto TEXT NOT NULL,
    marca TEXT,
    tipo_producto TEXT,
    presentacion TEXT,
    contenido_neto TEXT,
    unidad_medida TEXT,
    categoria_sugerida TEXT,
    codigo_barras TEXT UNIQUE,
    precio_venta REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_productos_nombre ON productos(nombre_producto);
"""


def _resolve_path(settings: Settings) -> Path:
    path = Path(settings.sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def init_db(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    path = _resolve_path(settings)
    key = str(path.resolve())
    with _lock:
        if key in _initialized:
            return path
        with sqlite3.connect(path) as conn:
            conn.executescript(SCHEMA)
            conn.commit()
        _initialized.add(key)
    return path


def get_connection(settings: Settings | None = None) -> sqlite3.Connection:
    settings = settings or get_settings()
    path = init_db(settings)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
