"""Descarga imágenes de productos peruanos desde Wikimedia Commons.

Uso: python scripts/fetch_commons_images.py
Lee la lista interna de marcas y guarda la mejor imagen de cada una en `demo/`.
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "TesisSprintMVP/0.1 (academic; christopher.flores.ri@gmail.com)"
THUMB_WIDTH = 900
DEST = Path("demo")
SLEEP = 0.4

# (slug_para_archivo, query_para_buscar_en_commons)
QUERIES: list[tuple[str, str]] = [
    ("inca_kola", "Inca Kola bottle"),
    ("kola_real", "Kola Real Peru"),
    ("big_cola", "Big Cola Peru"),
    ("cielo_agua", "Cielo agua mineral"),
    ("san_mateo_agua", "Agua San Mateo"),
    ("pisco_quebranta", "Pisco Quebranta bottle"),
    ("pisco_acholado", "Pisco acholado Peru"),
    ("cusquena_cerveza", "Cusquena beer bottle"),
    ("pilsen_callao", "Pilsen Callao beer"),
    ("cocinero_aceite", "Aceite Cocinero Peru"),
    ("primor_aceite", "Aceite Primor Peru"),
    ("alacena_mayonesa", "Mayonesa Alacena Peru"),
    ("quaker_avena", "Quaker Oats package"),
    ("don_vittorio_fideos", "Don Vittorio fideos"),
    ("nicolini_fideos", "Nicolini fideos Peru"),
    ("costeno_arroz", "Costeno arroz Peru"),
    ("paisana_arroz", "Paisana arroz"),
    ("gloria_leche", "Gloria leche evaporada"),
    ("laive_leche", "Laive leche Peru"),
    ("pura_vida_leche", "Pura Vida leche Peru"),
    ("bonle_leche", "Bonle leche Peru"),
    ("colgate_pasta", "Colgate toothpaste"),
    ("sensodyne_pasta", "Sensodyne toothpaste"),
    ("listerine_enjuague", "Listerine mouthwash"),
    ("rexona_desodorante", "Rexona deodorant"),
    ("axe_desodorante", "Axe deodorant"),
    ("old_spice_desodorante", "Old Spice deodorant stick"),
    ("speed_stick_desodorante", "Speed Stick deodorant"),
    ("nivea_creme", "Nivea Creme tin"),
    ("ponds_crema", "Pond's cream jar"),
    ("eucerin_crema", "Eucerin cream"),
    ("cicatricure", "Cicatricure cream"),
    ("hipoglos", "Hipoglos crema"),
    ("johnson_baby_shampoo", "Johnson's Baby Shampoo bottle"),
    ("huggies_panal", "Huggies diapers package"),
    ("pampers_panal", "Pampers diapers"),
    ("babysec_panal", "Babysec diapers"),
    ("always_toalla", "Always sanitary pads"),
    ("kotex_toalla", "Kotex sanitary pads"),
    ("nosotras_toalla", "Nosotras sanitary pads"),
    ("elite_papel", "Papel higienico Elite"),
    ("scott_papel", "Scott paper towel"),
    ("aspirina_bayer", "Aspirin Bayer tablets"),
    ("panadol", "Panadol tablets"),
    ("vicks_vaporub", "Vicks VapoRub jar"),
    ("centrum_multivitaminico", "Centrum multivitamin bottle"),
    ("sal_de_andrews", "Sal de Andrews"),
    ("alka_seltzer", "Alka Seltzer"),
    ("mentholatum", "Mentholatum jar"),
    ("ariel_detergente", "Ariel detergent"),
    ("skip_detergente", "Skip detergent box"),
    ("opal_detergente", "Detergente Opal Peru"),
    ("ace_detergente", "Ace detergent"),
    ("ayudin_lejia", "Ayudin lejia bottle"),
    ("clorox", "Clorox bottle"),
    ("poett_ambientador", "Poett desinfectante"),
    ("doritos", "Doritos bag"),
    ("pringles", "Pringles can"),
    ("oreo_galleta", "Oreo cookie package"),
    ("cua_cua", "Cua Cua chocolate"),
    ("sublime_chocolate", "Sublime chocolate Peru"),
    ("frugos_jugo", "Frugos jugo"),
    ("pulp_jugo", "Pulp jugo Peru"),
    ("sporade", "Sporade Peru"),
    ("powerade", "Powerade bottle"),
    ("gatorade", "Gatorade bottle"),
    ("red_bull", "Red Bull can"),
    ("volt_energizante", "Volt energy drink"),
]


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(c for c in normalized if not unicodedata.combining(c))
    ascii_value = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    return ascii_value or "image"


def http_json(params: dict[str, str]) -> dict:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_file(query: str) -> str | None:
    data = http_json(
        {
            "action": "query",
            "list": "search",
            "srnamespace": "6",
            "srsearch": query + " filetype:bitmap",
            "srlimit": "5",
            "format": "json",
        }
    )
    hits = data.get("query", {}).get("search", [])
    for hit in hits:
        title = hit["title"]
        # Filtra logos/SVG/iconos pequeños y categorías
        if title.lower().endswith(".svg"):
            continue
        if "logo" in title.lower():
            continue
        return title
    return None


def get_thumb_url(title: str) -> str | None:
    data = http_json(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "iiurlwidth": str(THUMB_WIDTH),
            "format": "json",
        }
    )
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    info = page.get("imageinfo", [])
    if not info:
        return None
    info0 = info[0]
    return info0.get("thumburl") or info0.get("url")


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 4 * 1024:
            return False
        dest.write_bytes(data)
        return True
    except Exception as exc:
        print(f"  ! download error: {exc}", file=sys.stderr)
        return False


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    ok = skipped = failed = 0
    for slug, query in QUERIES:
        target_jpg = DEST / f"{slug}.jpg"
        target_png = DEST / f"{slug}.png"
        if target_jpg.exists() or target_png.exists():
            skipped += 1
            continue
        title = search_file(query)
        if not title:
            print(f"[NF] {slug}: nada para '{query}'")
            failed += 1
            continue
        thumb = get_thumb_url(title)
        if not thumb:
            print(f"[NF] {slug}: no thumb para {title}")
            failed += 1
            continue
        ext = ".png" if thumb.lower().endswith(".png") else ".jpg"
        dest = DEST / f"{slug}{ext}"
        ok_dl = download(thumb, dest)
        if ok_dl:
            print(f"[OK] {slug:28s} <- {title}")
            ok += 1
        else:
            print(f"[ER] {slug:28s} <- {title}")
            failed += 1
        time.sleep(SLEEP)
    print(f"\nDescargadas: {ok} | Saltadas: {skipped} | Fallaron: {failed}")


if __name__ == "__main__":
    main()
