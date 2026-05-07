"""Segunda ronda con queries alternativas para marcas peruanas pendientes."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_commons_images import DEST, search_file, get_thumb_url, download, slugify, time, SLEEP


PENDING: list[tuple[str, list[str]]] = [
    ("vicks_vaporub", ["VapoRub", "Vicks VapoRub", "Mentholatum"]),
    ("mentholatum", ["Mentholatum cream"]),
    ("old_spice_desodorante", ["Old Spice", "Old Spice antiperspirant"]),
    ("huggies_panal", ["Huggies pack", "Huggies brand", "Pampers Huggies"]),
    ("babysec_panal", ["disposable diapers package", "diapers shelf"]),
    ("nosotras_toalla", ["sanitary pads package", "sanitary napkin pack"]),
    ("ayudin_lejia", ["bleach bottle"]),
    ("frugos_jugo", ["juice box pack", "tetra pak juice"]),
    ("pulp_jugo", ["nectar juice tetra"]),
    ("volt_energizante", ["energy drink can"]),
    ("skip_detergente", ["laundry detergent powder"]),
    ("opal_detergente", ["laundry detergent box"]),
    ("poett_ambientador", ["air freshener bottle"]),
    ("cocinero_aceite", ["sunflower oil bottle"]),
    ("primor_aceite", ["vegetable oil bottle"]),
    ("alacena_mayonesa", ["mayonnaise jar"]),
    ("nicolini_fideos", ["pasta package spaghetti"]),
    ("costeno_arroz", ["rice bag package"]),
    ("paisana_arroz", ["rice package"]),
    ("bonle_leche", ["milk evaporated can"]),
    ("cicatricure", ["scar cream tube"]),
    ("hipoglos", ["diaper rash cream tube"]),
]


def main() -> None:
    ok = failed = 0
    for slug, queries in PENDING:
        target_jpg = DEST / f"{slug}.jpg"
        target_png = DEST / f"{slug}.png"
        if target_jpg.exists() or target_png.exists():
            continue
        downloaded = False
        for query in queries:
            title = search_file(query)
            if not title:
                continue
            thumb = get_thumb_url(title)
            if not thumb:
                continue
            ext = ".png" if thumb.lower().endswith(".png") else ".jpg"
            dest = DEST / f"{slug}{ext}"
            if download(thumb, dest):
                print(f"[OK] {slug:28s} ({query!r}) <- {title}")
                ok += 1
                downloaded = True
                break
            time.sleep(SLEEP)
        if not downloaded:
            print(f"[NF] {slug}: no se encontró nada")
            failed += 1
        time.sleep(SLEEP)
    print(f"\nDescargadas: {ok} | Fallaron: {failed}")


if __name__ == "__main__":
    main()
