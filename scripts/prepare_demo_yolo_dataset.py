from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import unicodedata
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    ascii_value = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    return ascii_value or "image"


def find_images(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
    )


def write_dataset_yaml(output: Path, class_name: str) -> None:
    yaml_text = "\n".join(
        [
            f"path: {output.resolve()}",
            "train: images/train",
            "val: images/val",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    (output / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def prepare_dataset(source: Path, output: Path, val_ratio: float, seed: int, class_name: str) -> None:
    images = find_images(source)
    if not images:
        raise SystemExit(f"No se encontraron imágenes en {source}")

    random.Random(seed).shuffle(images)
    val_count = max(1, round(len(images) * val_ratio)) if len(images) > 1 else 0
    val_set = set(images[:val_count])

    if output.exists():
        shutil.rmtree(output)

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    used_names: set[str] = set()
    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] imagen omitida porque OpenCV no la pudo leer: {image_path}")
            continue

        split = "val" if image_path in val_set else "train"
        height, width = image.shape[:2]
        base_name = f"{index:04d}_{slugify(image_path.stem)}"
        while base_name in used_names:
            base_name = f"{base_name}_{index}"
        used_names.add(base_name)

        destination_image = output / "images" / split / f"{base_name}{image_path.suffix.lower()}"
        destination_label = output / "labels" / split / f"{base_name}.txt"

        shutil.copy2(image_path, destination_image)
        destination_label.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")

        rows.append(
            {
                "source": str(image_path),
                "image": str(destination_image),
                "label": str(destination_label),
                "split": split,
                "width": width,
                "height": height,
                "label_mode": "full_image_bbox",
            }
        )

    write_dataset_yaml(output, class_name)
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "image", "label", "split", "width", "height", "label_mode"],
        )
        writer.writeheader()
        writer.writerows(rows)

    train_count = sum(1 for row in rows if row["split"] == "train")
    actual_val_count = sum(1 for row in rows if row["split"] == "val")
    print(f"Dataset YOLO creado en: {output}")
    print(f"Imágenes: {len(rows)} | train: {train_count} | val: {actual_val_count}")
    print("Modo de etiqueta: caja completa por imagen para clase 0 product_label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara las imágenes demo como dataset YOLO con cajas completas."
    )
    parser.add_argument("--source", type=Path, default=Path("demo"), help="Carpeta con imágenes demo.")
    parser.add_argument("--output", type=Path, default=Path("data/demo_yolo"), help="Salida YOLO.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Proporción para validación.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para split reproducible.")
    parser.add_argument("--class-name", default="product_label", help="Nombre de clase YOLO.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio debe estar entre 0 y menor que 1")
    prepare_dataset(args.source, args.output, args.val_ratio, args.seed, args.class_name)


if __name__ == "__main__":
    main()

