"""Génère le dataset synthétique : images de cartes + nom de chaque carte et boîtes de ses deux coins.

Usage :
    python -m cards.dataset.generate_dataset --n-train 20000 --n-val 2000 --n-test 2000

Structure produite, lisible à la fois par notre code et par Ultralytics (les images ne sont écrites qu'une fois) :
    data/synthetic/data.yaml                          chemins des splits et noms des classes, pour Ultralytics
    data/synthetic/{train,val,test}/images/000000.jpg
    data/synthetic/{train,val,test}/labels/000000.txt étiquettes YOLO : une ligne par coin visible
    data/synthetic/{train,val,test}/annotations.json  annotations complètes : cartes, coins, visibilité
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..deck import DECK
from .scene import MIN_CORNER_VISIBILITY, load_decks, make_scene

IMAGE_SIZE = 384


def generate_split(out_dir, n_images, decks, seed):
    rng = np.random.default_rng(seed)  # seed fixe par split : le dataset est reproductible
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(n_images):
        image, cards, corners, deck_name = make_scene(decks, rng, IMAGE_SIZE)
        image_path = f"images/{i:06d}.jpg"
        image.save(out_dir / image_path, quality=92)
        item = {
            "image": image_path,
            "deck": deck_name,
            "cards": [
                {
                    "name": card.name,
                    "class_id": card.class_id,
                    "corners": [{**corner, "visible": corner["visibility"] >= MIN_CORNER_VISIBILITY} for corner in card_corners],
                }
                for card, card_corners in zip(cards, corners)
            ],
        }
        (out_dir / "labels" / f"{i:06d}.txt").write_text(yolo_labels(item))
        items.append(item)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_images}")

    annotations = {"image_size": IMAGE_SIZE, "min_corner_visibility": MIN_CORNER_VISIBILITY, "items": items}
    (out_dir / "annotations.json").write_text(json.dumps(annotations))
    n_cards = sum(len(item["cards"]) for item in items)
    n_visible = sum(c["visible"] for item in items for card in item["cards"] for c in card["corners"])
    print(f"{out_dir}: {n_images} images, {n_cards} cartes, {n_visible} coins visibles sur {2 * n_cards}")


def yolo_labels(item, size=IMAGE_SIZE):
    """Étiquettes YOLO d'une image : une ligne « class_id cx cy w h » (entre 0 et 1) par coin visible.

    Les coins trop cachés ne sont pas écrits : YOLO n'a pas de notion de « boîte à ignorer ».
    Une image sans carte donne un fichier vide, c'est la convention YOLO.
    """
    lines = []
    for card in item["cards"]:
        for corner in card["corners"]:
            if corner["visible"]:
                x1, y1, x2, y2 = corner["box"]
                cx, cy, w, h = (x1 + x2) / 2 / size, (y1 + y2) / 2 / size, (x2 - x1) / size, (y2 - y1) / size
                lines.append(f"{card['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return "".join(lines)


def write_data_yaml(out_dir):
    """data.yaml lu par Ultralytics : chemins des trois splits et noms des 52 classes.

    Ultralytics trouve les étiquettes d'une image en remplaçant /images/ par /labels/ dans son chemin.
    """
    names = "\n".join(f"  {card.class_id}: {card.name}" for card in DECK)
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n{names}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Génère le dataset synthétique de cartes.")
    parser.add_argument("--out-dir", default="data/synthetic")
    parser.add_argument("--cards-dir", default="data/cards")
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    decks = load_decks(args.cards_dir)
    for offset, (split, n_images) in enumerate([("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]):
        print(f"Génération de {split}...")
        generate_split(out_dir / split, n_images, decks, args.seed + offset)
    write_data_yaml(out_dir)


if __name__ == "__main__":
    main()
