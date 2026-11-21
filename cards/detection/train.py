"""Fine-tune un YOLO pré-entraîné (COCO) sur le dataset (data/synthetic/data.yaml) : un objet par coin visible, 52 classes.

Usage :
    python -m cards.detection.train                                  # yolov8s, 50 epochs
    python -m cards.detection.train --model yolov8n.pt --epochs 3    # essai rapide
    python -m cards.detection.train --backup-dir /content/drive/MyDrive/adl3a_runs/yolov8s_checkpoints --resume   # Colab

Les poids sont écrits dans runs/detect/<nom>/weights/best.pt.

Sur Colab, le disque est effacé quand la session s'arrête (limite de GPU, coupure du navigateur) :
--backup-dir copie last.pt, best.pt et results.csv ailleurs (Google Drive) à chaque epoch, et --resume
reprend l'entraînement à la dernière epoch sauvegardée. Sans sauvegarde existante, --resume démarre
un entraînement neuf, et si l'entraînement est déjà terminé il ne fait rien : la même commande sert pour
le premier lancement, pour les reprises et pour une cellule relancée par erreur.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

IMAGE_SIZE = 384  # la taille des images du dataset
BACKED_UP = ("weights/last.pt", "weights/best.pt", "results.csv")


def backup_callback(backup_dir):
    """Copie les fichiers utiles dans backup_dir juste après chaque sauvegarde du modèle."""

    def backup(trainer):
        backup_dir.mkdir(parents=True, exist_ok=True)
        for name in BACKED_UP:
            source = trainer.save_dir / name
            try:
                if source.exists():
                    shutil.copy(source, backup_dir / Path(name).name)
            except OSError as error:  # un souci passager de Drive ne doit pas interrompre l'entraînement
                print(f"Sauvegarde de {name} impossible : {error}")

    return backup


def is_finished(last):
    """Un entraînement terminé (ou arrêté par la patience) laisse un last.pt allégé, dont l'epoch vaut -1."""
    import torch

    return torch.load(last, map_location="cpu", weights_only=False).get("epoch") == -1


def restore_last(backup_dir, last):
    """Remet last.pt à sa place depuis la sauvegarde, sur une machine neuve. True si last.pt existe ensuite."""
    saved = backup_dir / "last.pt" if backup_dir else None
    if not last.exists() and saved is not None and saved.exists():
        last.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(saved, last)
        results = backup_dir / "results.csv"
        if results.exists():
            shutil.copy(results, last.parent.parent / "results.csv")
        print(f"Reprise depuis la sauvegarde {saved}")
    return last.exists()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO sur les coins des cartes.")
    parser.add_argument("--data", default="data/synthetic/data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="poids de départ, pré-entraînés sur COCO")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="mps", help="mps (GPU du Mac), cuda ou cpu")
    parser.add_argument("--patience", type=int, default=10, help="arrêt si la validation ne progresse plus pendant N epochs")
    parser.add_argument("--name", default=None, help="nom du dossier dans runs/detect/ (défaut : nom du modèle)")
    parser.add_argument("--backup-dir", default=None, help="copie last.pt, best.pt et results.csv ici à chaque epoch")
    parser.add_argument("--resume", action="store_true", help="reprend depuis last.pt (ou la sauvegarde) s'il existe")
    args = parser.parse_args()

    project = Path("runs/detect").resolve()
    name = args.name or Path(args.model).stem
    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    last = project / name / "weights" / "last.pt"

    if args.resume and restore_last(backup_dir, last):
        if is_finished(last):
            print(f"L'entraînement est déjà terminé, rien à reprendre. Poids : {backup_dir or last.parent}/best.pt")
            return
        model = YOLO(last)
        resume = True
        print(f"Reprise de l'entraînement : {last}")
    else:
        if args.resume:
            print("Aucune sauvegarde à reprendre : entraînement neuf.")
        model = YOLO(args.model)
        resume = False
    if backup_dir:
        model.add_callback("on_model_save", backup_callback(backup_dir))  # après chaque epoch
        model.add_callback("on_train_end", backup_callback(backup_dir))  # les poids finaux, allégés

    if resume:
        # Les réglages de l'entraînement d'origine sont relus dans last.pt ; on ne peut changer que quelques options
        model.train(resume=True, data=args.data, device=args.device, batch=args.batch, patience=args.patience)
        return
    model.train(
        data=args.data,
        imgsz=IMAGE_SIZE,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        fliplr=0.0,  # une carte vue en miroir n'existe pas : pas de retournement horizontal
        patience=args.patience,  # arrêt anticipé : la validation plafonne avant la fin, inutile d'aller au bout
        project=str(project),
        name=name,
    )


if __name__ == "__main__":
    main()
