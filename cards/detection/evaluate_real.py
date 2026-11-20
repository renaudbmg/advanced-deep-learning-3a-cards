"""Évalue le détecteur sur de vraies photos (test hors distribution : entraînement uniquement synthétique).

Usage, après python -m cards.detection.real_dataset :
    python -m cards.detection.evaluate_real --weights checkpoints/yolov8s_ft35.pt --data-dir data/real/poker-cards/validation
    python -m cards.detection.evaluate_real --weights checkpoints/yolov8s_ft35.pt   # test, pour le chiffre final

Les réglages (--conf, --imgsz) se choisissent sur validation ; test ne sert qu'au chiffre final.

Les boîtes du dataset couvrent la carte entière, les nôtres couvrent les coins : on ne compare
donc pas les boîtes par IoU, sauf à titre indicatif pour les cartes dont les deux coins sont détectés.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw

from ..deck import DECK
from .evaluate import iou
from .predict import IMAGE_SIZE, center, detect, draw_scene, label_font

RELATION_MARGIN = 0.25  # une relation n'est comptée que si l'écart réel dépasse 1/4 de la taille moyenne des deux cartes


def inside(point, box):
    return box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]


def evaluate_real_image(scene, item):
    """Compare la scène prédite à la vérité d'une photo. Retourne le détail, pour cumuler et pour results.json."""
    truth = {card["class_id"]: card["box"] for card in item["cards"]}
    predicted = {card["class_id"]: card for card in scene["cards"]}
    found = [cid for cid in predicted if cid in truth]
    ghosts = [cid for cid in predicted if cid not in truth]
    missed = [cid for cid in truth if cid not in predicted]
    # Bien localisée : un coin détecté a son centre dans la boîte réelle de la carte
    localized = [cid for cid in found if any(inside(center(corner), truth[cid]) for corner in predicted[cid]["corners"])]

    # Relations spatiales entre paires de cartes bien identifiées : même ordre gauche/droite et haut/bas ?
    relations = {"horizontal": [0, 0], "vertical": [0, 0]}  # [justes, comptées]
    for a, b in combinations(found, 2):
        ta, tb = truth[a], truth[b]
        (tax, tay), (tbx, tby) = center(ta), center(tb)
        (pax, pay), (pbx, pby) = center(predicted[a]["box"]), center(predicted[b]["box"])
        width = ((ta[2] - ta[0]) + (tb[2] - tb[0])) / 2
        height = ((ta[3] - ta[1]) + (tb[3] - tb[1])) / 2
        for axis, true_gap, pred_gap, size in (("horizontal", tax - tbx, pax - pbx, width), ("vertical", tay - tby, pay - pby, height)):
            if abs(true_gap) < RELATION_MARGIN * size:
                continue  # cartes presque alignées : la relation est ambiguë
            relations[axis][1] += 1
            relations[axis][0] += (true_gap > 0) == (pred_gap > 0)

    # IoU indicative : seulement les cartes dont les deux coins sont détectés (leur union ~ la carte)
    ious = [iou(predicted[cid]["box"], truth[cid]) for cid in found if len(predicted[cid]["corners"]) == 2]

    return {
        "found": found,
        "ghosts": ghosts,
        "missed": missed,
        "localized": localized,
        "relations": relations,
        "ious": ious,
        "exact": not ghosts and not missed,
    }


def draw_errors(image, scene, item, result):
    """Prédictions (vert/jaune), boîtes réelles (bleu), cartes manquées et fantômes écrites en rouge."""
    image = draw_scene(image, scene)
    draw = ImageDraw.Draw(image)
    for card in item["cards"]:
        draw.rectangle([card["box"][0], card["box"][1], card["box"][2] - 1, card["box"][3] - 1], outline="blue", width=2)
    font = label_font(14)
    lines = []
    if result["missed"]:
        lines.append("manquées : " + ", ".join(DECK[cid].name for cid in result["missed"]))
    if result["ghosts"]:
        lines.append("fantômes : " + ", ".join(DECK[cid].name for cid in result["ghosts"]))
    unlocalized = [cid for cid in result["found"] if cid not in result["localized"]]
    if unlocalized:
        lines.append("mal placées : " + ", ".join(DECK[cid].name for cid in unlocalized))
    for i, line in enumerate(lines):
        position = (4, 4 + i * 18)
        draw.rectangle(draw.textbbox(position, line, font=font), fill="white")
        draw.text(position, line, fill="red", font=font)
    return image


def ratio(a, b):
    return f"{a / b:.3f} ({a}/{b})" if b else "n/a (0 cas)"


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Évalue le détecteur sur de vraies photos de cartes.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="data/real/poker-cards/test")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--out-dir", default=None, help="défaut : runs/real_eval/<poids>/<split>_<imgsz>/")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    items = json.loads((data_dir / "annotations.json").read_text())["items"]
    out_dir = Path(args.out_dir or f"runs/real_eval/{Path(args.weights).stem}/{data_dir.name}_{args.imgsz}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    total = {"found": 0, "ghosts": 0, "missed": 0, "localized": 0, "exact": 0}
    relations = {"horizontal": [0, 0], "vertical": [0, 0]}
    ious, details = [], []
    for item in items:
        image = Image.open(data_dir / item["image"]).convert("RGB")
        scene = detect(model, image, args.conf, args.imgsz)
        result = evaluate_real_image(scene, item)
        for key in ("found", "ghosts", "missed", "localized"):
            total[key] += len(result[key])
        total["exact"] += result["exact"]
        for axis in relations:
            relations[axis][0] += result["relations"][axis][0]
            relations[axis][1] += result["relations"][axis][1]
        ious += result["ious"]
        details.append({"image": item["image"], **{k: v for k, v in result.items() if k != "ious"}, "scene": scene})
        if not result["exact"] or len(result["localized"]) < len(result["found"]):
            draw_errors(image, scene, item, result).save(out_dir / Path(item["image"]).name)

    (out_dir / "results.json").write_text(json.dumps(details))
    tp, fp, fn = total["found"], total["ghosts"], total["missed"]
    print(f"{len(items)} photos, {tp + fn} cartes ({data_dir}, conf {args.conf}, imgsz {args.imgsz})")
    print(f"Identification : précision {ratio(tp, tp + fp)}, rappel {ratio(tp, tp + fn)}")
    print(f"Cartes fantômes : {fp}, cartes manquées : {fn}")
    print(f"Images exactes : {ratio(total['exact'], len(items))}")
    print(f"Cartes bien localisées (parmi les bien identifiées) : {ratio(total['localized'], tp)}")
    print(f"Relations gauche/droite justes : {ratio(*relations['horizontal'])}")
    print(f"Relations haut/bas justes : {ratio(*relations['vertical'])}")
    if ious:
        good = sum(v >= 0.5 for v in ious)
        print(f"IoU des cartes à deux coins ({len(ious)} cartes) : moyenne {sum(ious) / len(ious):.3f}, part >= 0,5 : {ratio(good, len(ious))}")
    print(f"Images des erreurs et détail : {out_dir}/")


if __name__ == "__main__":
    main()
