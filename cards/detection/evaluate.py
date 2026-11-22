"""Évalue la détection par carte, après regroupement des coins, sur un split du dataset.

Usage :
    python -m cards.detection.evaluate --weights runs/detect/yolov8s/weights/best.pt
    python -m cards.detection.evaluate --weights ... --map   # ajoute la mAP par coin d'Ultralytics

Une carte prédite est correcte si la vérité terrain contient une carte de même class_id et
qu'au moins un coin prédit a une IoU >= 0,5 avec l'un de ses deux coins.
"""

import argparse
import json
from pathlib import Path

from .predict import detect

IOU_THRESHOLD = 0.5


def iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return inter / (area(a) + area(b) - inter) if inter else 0.0


def touches(pred_corners, true_corners):
    return any(iou(p, t) >= IOU_THRESHOLD for p in pred_corners for t in true_corners)


def evaluate_image(scene, item):
    """Compte les vrais positifs, faux positifs, cartes manquées et cartes en double d'une image."""
    truth = {card["class_id"]: [c["box"] for c in card["corners"]] for card in item["cards"]}
    found = {p["class_id"] for p in scene["cards"] if p["class_id"] in truth and touches(p["corners"], truth[p["class_id"]])}
    false_positives = [p for p in scene["cards"] if p["class_id"] not in found]
    # Carte en double : une fausse carte posée sur une vraie carte déjà trouvée (ses deux coins ont reçu deux classes)
    duplicates = sum(any(touches(p["corners"], truth[c]) for c in found) for p in false_positives)
    exact = {p["class_id"] for p in scene["cards"]} == set(truth)
    return len(found), len(false_positives), len(truth) - len(found), duplicates, exact


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Évalue la détection des cartes.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split-dir", default="data/synthetic/test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--map", action="store_true", help="calcule aussi la mAP par coin")
    parser.add_argument("--yolo-data", default="data/synthetic/data.yaml")
    args = parser.parse_args()

    model = YOLO(args.weights)
    split_dir = Path(args.split_dir)
    items = json.loads((split_dir / "annotations.json").read_text())["items"]

    tp = fp = fn = duplicates = exact = 0
    for item in items:
        scene = detect(model, split_dir / item["image"], args.conf)
        t, f, n, d, e = evaluate_image(scene, item)
        tp, fp, fn, duplicates, exact = tp + t, fp + f, fn + n, duplicates + d, exact + e

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"{len(items)} images, {tp + fn} cartes")
    print(f"Par carte : précision {precision:.3f}, rappel {recall:.3f} ({tp} trouvées, {fp} fausses, {fn} manquées)")
    print(f"Cartes en double : {duplicates}")
    print(f"Images exactes (ensemble des cartes prédit = attendu) : {exact / len(items):.3f}")

    if args.map:
        # project explicite : sinon Ultralytics écrit dans le runs_dir de ses réglages globaux, hors du repo
        metrics = model.val(
            data=args.yolo_data, split="test", imgsz=384, verbose=False, project=str(Path("runs/detect").resolve()), name="val"
        )
        print(f"Par coin : mAP50 {metrics.box.map50:.3f}, mAP50-95 {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
