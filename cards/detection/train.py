"""Fine-tune un YOLO pré-entraîné (COCO) sur le dataset (data/synthetic/data.yaml) : un objet par coin visible, 52 classes.

Usage :
    python -m cards.detection.train                                  # yolov8s, 50 epochs
    python -m cards.detection.train --model yolov8n.pt --epochs 3    # essai rapide

Les poids sont écrits dans runs/detect/<nom>/weights/best.pt.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

IMAGE_SIZE = 384  # la taille des images du dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO sur les coins des cartes.")
    parser.add_argument("--data", default="data/synthetic/data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="poids de départ, pré-entraînés sur COCO")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="mps", help="mps (GPU du Mac), cuda ou cpu")
    parser.add_argument("--patience", type=int, default=10, help="arrêt si la validation ne progresse plus pendant N epochs")
    parser.add_argument("--name", default=None, help="nom du dossier dans runs/detect/ (défaut : nom du modèle)")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=IMAGE_SIZE,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        fliplr=0.0,  # une carte vue en miroir n'existe pas : pas de retournement horizontal
        patience=args.patience,  # le premier entraînement plafonnait vers l'epoch 25 : inutile d'aller au bout
        project=str(Path("runs/detect").resolve()),
        name=args.name or Path(args.model).stem,
    )


if __name__ == "__main__":
    main()
