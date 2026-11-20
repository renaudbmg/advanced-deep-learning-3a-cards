"""Télécharge un dataset de vraies photos de cartes, pour tester le détecteur hors distribution.

Source : Francesco/poker-cards-cxcvz sur Hugging Face (Roboflow 100), de vraies photos 640×640
avec 4 à 5 cartes chacune. Ces images ne servent qu'à évaluer : ni à entraîner, ni à régler sur le test.

Usage :
    python -m cards.detection.real_dataset --split test         # 193 images, pour le chiffre final
    python -m cards.detection.real_dataset --split validation   # 128 images, pour régler --conf et --imgsz

Produit data/real/poker-cards/<split>/images/*.jpg et annotations.json :
    {"source": ..., "license": ..., "items": [{"image": "images/116.jpg",
      "cards": [{"name": "10 of diamonds", "class_id": 8, "box": [x1, y1, x2, y2]}]}]}
Les boîtes couvrent la carte entière (pas les coins comme notre dataset) : x2 et y2 exclusifs.
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from ..deck import DECK

DATASET = "Francesco/poker-cards-cxcvz"
SOURCE = f"https://huggingface.co/datasets/{DATASET}"
LICENSE = "CC BY 4.0 (Roboflow 100), à vérifier sur la page d'origine"
API = "https://datasets-server.huggingface.co"
USER_AGENT = "playing-cards-eval/0.1"
PAGE = 50  # lignes par requête

# Leurs noms -> les nôtres : « 10 Diamonds » -> « 10 of diamonds », « A Trefoils » -> « ace of clubs »
RANKS = {"A": "ace", "J": "jack", "Q": "queen", "K": "king"}
SUITS = {"Diamonds": "diamonds", "Hearts": "hearts", "Spades": "spades", "Trefoils": "clubs"}
CLASS_IDS = {card.name: card.class_id for card in DECK}


def http_get(url, tries=6):
    """GET avec quelques essais : le serveur renvoie parfois une erreur 502 passagère."""
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def api(endpoint, **params):
    return json.loads(http_get(f"{API}/{endpoint}?{urllib.parse.urlencode({'dataset': DATASET, 'config': 'default', **params})}"))


def true_category_name(category, their_names):
    """Nom réel de la carte annotée avec ce numéro de classe ; None si ce n'est pas une carte.

    La liste de noms du dataset ne correspond pas aux numéros pour les classes 1 à 21 (erreur
    présente dans le fichier COCO d'origine de Roboflow, pas seulement sur Hugging Face) :
    - classes 1 à 19 : la vraie carte est le nom situé deux crans plus loin dans la liste
      (la classe « 59 », n° 1, est le 10 de cœur ; n° 2, « 10 Diamonds », est le 10 de pique) ;
    - classe 21 : le 10 de carreau ;
    - classes 0 (« poker-cards ») et 20 : jamais utilisées ;
    - classes 22 à 53 : le nom de la liste est juste.
    Correspondance établie en recadrant les boîtes de chaque classe et en regardant les cartes.
    """
    if 1 <= category <= 19:
        return their_names[category + 2]
    if category == 21:
        return "10 Diamonds"
    if category >= 22:
        return their_names[category]
    return None


def our_name(their_name):
    """« Q Spades » -> « queen of spades » ; None pour les classes qui ne sont pas des cartes."""
    if their_name is None:
        return None
    parts = str(their_name).split()
    if len(parts) != 2 or parts[1] not in SUITS:
        return None
    rank = RANKS.get(parts[0], parts[0])
    name = f"{rank} of {SUITS[parts[1]]}"
    return name if name in CLASS_IDS else None


def main():
    parser = argparse.ArgumentParser(description="Télécharge les vraies photos de cartes.")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--out-dir", default="data/real/poker-cards")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.split
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    features = api("first-rows", split=args.split)["features"]
    objects = next(f for f in features if f["name"] == "objects")["type"]
    their_names = objects["feature"]["category"]["names"]
    n_rows = next(s["num_rows"] for s in api("size")["size"]["splits"] if s["split"] == args.split)

    items, skipped = [], Counter()
    for offset in range(0, n_rows, PAGE):
        for row in api("rows", split=args.split, offset=offset, length=PAGE)["rows"]:
            row = row["row"]
            image_path = f"images/{row['image_id']}.jpg"
            if not (out_dir / image_path).exists():  # relancer le script ne retélécharge pas les images
                (out_dir / image_path).write_bytes(http_get(row["image"]["src"]))
            cards = []
            for category, (x, y, w, h) in zip(row["objects"]["category"], row["objects"]["bbox"]):
                name = our_name(true_category_name(category, their_names))
                if name is None:
                    skipped[f"classe {category}"] += 1
                    continue
                cards.append({"name": name, "class_id": CLASS_IDS[name], "box": [round(x), round(y), round(x + w), round(y + h)]})
            items.append({"image": image_path, "cards": cards})
        print(f"  {len(items)}/{n_rows}")

    annotations = {"source": SOURCE, "license": LICENSE, "split": args.split, "items": items}
    (out_dir / "annotations.json").write_text(json.dumps(annotations))
    n_cards = sum(len(item["cards"]) for item in items)
    distinct = len({card["class_id"] for item in items for card in item["cards"]})
    print(f"{out_dir}: {len(items)} images, {n_cards} cartes, {distinct} cartes distinctes sur 52")
    print(f"Objets écartés (pas des cartes) : {dict(skipped) or 'aucun'}")


if __name__ == "__main__":
    main()
