"""Image -> scène : YOLO détecte les coins, group_corners() les regroupe en cartes.

Usage :
    python -m cards.detection.predict image.jpg --weights runs/detect/yolov8s/weights/best.pt

Affiche la scène en JSON et écrit une image annotée pour vérifier à l'œil.

Format de la scène :
    {"image_size": [w, h],
     "cards": [{"name": "queen of hearts", "class_id": 10, "box": [x1, y1, x2, y2],
                "corners": [[x1, y1, x2, y2], ...], "confidence": 0.94}]}
"""

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw

from ..dataset.scene import CARD_SIZE, CORNER_TOP_LEFT
from ..deck import DECK

IMAGE_SIZE = 384


def corner_distance_ratio():
    """Distance entre les centres des deux coins d'une carte, en hauteurs de coin (environ 3,7)."""
    x1, y1, x2, y2 = CORNER_TOP_LEFT
    width, height = CARD_SIZE
    # Centre du coin haut-gauche : ((x1 + x2) / 2, (y1 + y2) / 2) ; le coin bas-droit est son symétrique
    return math.hypot(width - (x1 + x2), height - (y1 + y2)) / (y2 - y1)


# Deux coins de même classe sont la même carte si leur distance tombe dans cet intervalle,
# en hauteurs de coin. La marge couvre la perspective, la rotation et l'imprécision des boîtes.
PAIR_DISTANCE = (0.7 * corner_distance_ratio(), 1.35 * corner_distance_ratio())


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def compatible(a, b):
    """Les deux détections peuvent-elles être les deux coins d'une même carte ?"""
    (ax, ay), (bx, by) = center(a["box"]), center(b["box"])
    corner_height = ((a["box"][3] - a["box"][1]) + (b["box"][3] - b["box"][1])) / 2
    distance = math.hypot(ax - bx, ay - by) / corner_height
    return PAIR_DISTANCE[0] <= distance <= PAIR_DISTANCE[1]


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def group_corners(detections):
    """Regroupe les coins détectés en cartes. Python pur, aucun paramètre appris.

    detections : [{"class_id": int, "box": [x1, y1, x2, y2], "confidence": float}, ...]
    Retourne (cartes, coins écartés).

    Une carte apparaît au plus une fois par image : un class_id = une carte. Pour chaque classe :
    - si deux coins sont à une distance compatible avec une carte, on garde la paire la plus confiante ;
    - sinon, on garde le coin le plus confiant seul.
    Les autres coins de la classe sont des erreurs de classe : ils sont écartés.
    """
    by_class = defaultdict(list)
    for detection in detections:
        by_class[detection["class_id"]].append(detection)

    cards, discarded = [], []
    for class_id, candidates in sorted(by_class.items()):
        pairs = [pair for pair in combinations(candidates, 2) if compatible(*pair)]
        if pairs:
            kept = list(max(pairs, key=lambda pair: pair[0]["confidence"] + pair[1]["confidence"]))
        else:
            kept = [max(candidates, key=lambda d: d["confidence"])]
        discarded += [d for d in candidates if all(d is not k for k in kept)]
        cards.append(
            {
                "name": DECK[class_id].name,
                "class_id": class_id,
                "box": union([d["box"] for d in kept]),  # 2 coins en diagonale : à peu près la carte entière
                "corners": [d["box"] for d in kept],
                "confidence": round(max(d["confidence"] for d in kept), 3),
            }
        )
    return cards, discarded


def detect(model, image, conf=0.25):
    """Image (chemin ou PIL) -> scène."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    # agnostic_nms : si deux classes sont prédites sur le même coin, seule la plus confiante reste.
    # Les deux coins d'une même carte ne se recouvrent pas, ils survivent tous les deux.
    result = model.predict(image, imgsz=IMAGE_SIZE, conf=conf, agnostic_nms=True, verbose=False)[0]
    detections = [
        {"class_id": int(c), "box": [round(v) for v in box], "confidence": float(p)}
        for box, c, p in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
    ]
    cards, _ = group_corners(detections)
    return {"image_size": list(image.size), "cards": cards}


def draw_scene(image, scene):
    """Dessine les coins (vert), la boîte de chaque carte (jaune) et son nom."""
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for card in scene["cards"]:
        draw.rectangle([card["box"][0], card["box"][1], card["box"][2] - 1, card["box"][3] - 1], outline="yellow")
        for corner in card["corners"]:
            draw.rectangle([corner[0], corner[1], corner[2] - 1, corner[3] - 1], outline="lime", width=2)
        x1, y1 = card["corners"][0][:2]
        label = f"{card['name']} {card['confidence']:.2f}"
        draw.rectangle(draw.textbbox((x1, y1 - 11), label), fill="lime")
        draw.text((x1, y1 - 11), label, fill="black")
    return image


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Détecte les cartes d'une image.")
    parser.add_argument("image")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, default=0.25, help="confiance minimale d'un coin")
    parser.add_argument("--out", default=None, help="image annotée (défaut : runs/predict/<nom>.jpg)")
    args = parser.parse_args()

    scene = detect(YOLO(args.weights), args.image, args.conf)
    print(json.dumps(scene, indent=2))
    out = Path(args.out or f"runs/predict/{Path(args.image).stem}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    draw_scene(Image.open(args.image), scene).save(out)
    print(f"Image annotée : {out}")


if __name__ == "__main__":
    main()
