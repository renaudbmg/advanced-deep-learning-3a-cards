# Détection

Un YOLO pré-entraîné, fine-tuné sur le dataset synthétique ([../dataset/](../dataset/)), détecte les **coins** des cartes. Une fonction Python regroupe ensuite ces coins en **cartes**. En sortie, on obtient une **scène** : la liste des cartes présentes, avec leurs positions.

## Utilisation

Depuis la racine du repo, après `python -m cards.dataset.generate_dataset` :

```bash
python -m cards.detection.train                                   # fine-tuning de yolov8s, 50 epochs
python -m cards.detection.predict image.jpg --weights runs/detect/yolov8s/weights/best.pt
python -m cards.detection.evaluate --weights runs/detect/yolov8s/weights/best.pt --map # map = mean avergage precision
```

| Commande | Options principales |
|---|---|
| `train` | `--model` (défaut `yolov8s.pt`), `--epochs` (50), `--batch` (32), `--device` (`mps`), `--data`, `--name` |
| `predict` | `--weights`, `--conf` (0,25), `--out` (défaut `runs/predict/<nom>.jpg`) |
| `evaluate` | `--weights`, `--split-dir` (défaut `data/synthetic/test`), `--conf`, `--map` (ajoute la mAP par coin) |

## Entraîner sur Colab

Sur un MacBook M1 de 8 Go, l'entraînement complet (yolov8s, 50 epochs, 20 000 images) prendrait plusieurs dizaines d'heures : environ 160 s pour 1 000 images à `--batch 16` en essai, et `--batch 32` est trop lourd pour la mémoire. On l'envoie donc sur le GPU gratuit de Colab avec [colab_training.ipynb](colab_training.ipynb).

Le notebook clone le repo depuis GitHub (branche réglable en tête de notebook), régénère le dataset sur place (la seed fixe donne le même qu'en local), entraîne avec `--device 0`, puis copie les résultats sur Google Drive.

**Le code doit être poussé sur GitHub** : Colab clone la branche, il ne voit pas les changements locaux. Pour ouvrir le notebook : *Fichier → Ouvrir le notebook → GitHub*, coller l'URL du repo, choisir la branche, puis `cards/detection/colab_training.ipynb`.

Une fois `best.pt` récupéré, l'évaluation se fait en local avec `evaluate`.

## Les fichiers

| Fichier | Rôle |
|---|---|
| [train.py](train.py) | Fine-tuning avec Ultralytics. Les poids sont écrits dans `runs/detect/<nom>/weights/best.pt`. |
| [predict.py](predict.py) | `detect(model, image)` → scène, `group_corners(detections)` → cartes, `draw_scene(image, scene)` → image annotée. |
| [evaluate.py](evaluate.py) | Métriques par carte sur un split du dataset, et mAP par coin. |

## Format de la scène

```json
{
  "image_size": [384, 384],
  "cards": [
    {"name": "queen of hearts", "class_id": 10, "box": [62, 57, 252, 257],
     "corners": [[62, 57, 98, 116], [215, 199, 252, 257]], "confidence": 0.94}
  ]
}
```

- `corners` : les boîtes des coins retenus pour cette carte, 1 ou 2, en `[x1, y1, x2, y2]` pixels.
- `box` : l'union des coins. Avec deux coins, opposés en diagonale, elle recouvre à peu près la carte entière. Avec un seul coin, c'est la boîte de ce coin.
- `confidence` : la plus haute confiance parmi les coins retenus.

## Décisions

### Le modèle

| Décision | Valeur | Pourquoi |
|---|---|---|
| Modèle | `yolov8s`, pré-entraîné sur COCO, avec Ultralytics | La tâche est simple (images synthétiques, un seul design de cartes) : pas besoin d'un gros modèle. Les tailles n et s s'entraînent sur un portable. Ultralytics lit directement le dossier du dataset (`data/synthetic/data.yaml`). Voir [docs/useful_links.md](../../docs/useful_links.md) (revue YOLO v1-v8). |
| Classes | 52, une par carte (`class_id` de [../deck.py](../deck.py)) | YOLO reconnaît la carte en même temps qu'il la localise. |
| Objet détecté | Un **coin visible** | Voir le README du dataset : le coin reste visible quand les cartes se recouvrent, et contient le rang et l'enseigne. |
| Taille d'entrée | 384 px | La taille des images du dataset. La tête la plus fine de YOLOv8 travaille au stride 8 (48×48 cases), ce qui suffit pour des coins d'environ 10×19 px. |
| Retournement horizontal | Désactivé (`fliplr=0.0`) | Ultralytics retourne par défaut une image sur deux en miroir. Une carte en miroir n'existe pas : le modèle apprendrait des coins impossibles. |
| Autres augmentations | Celles d'Ultralytics par défaut (mosaïque, HSV, échelle) | Elles complètent celles du générateur. |

### Des coins aux cartes : `group_corners()`

C'est du **Python pur**, sans paramètre appris : un post-traitement, comme la NMS.

| Cas | Traitement | Pourquoi |
|---|---|---|
| Règle de base | **Un `class_id` = une carte** | Une carte apparaît au plus une fois par image : deux coins de même classe sont la même carte. |
| Deux classes prédites sur le même coin (6 ou 9, cœur ou carreau) | NMS **agnostique** (`agnostic_nms=True`) : seule la plus confiante reste | Les deux coins d'une même carte ne se recouvrent pas : ils ne s'éliminent pas l'un l'autre. |
| Plus de deux coins d'une même classe, ou deux coins trop loin l'un de l'autre | On garde la paire de coins **compatibles** la plus confiante. S'il n'y en a pas, on garde le coin le plus confiant. Les autres sont écartés. | Un coin en trop vient d'une erreur de classe sur une autre carte. Deux coins à la bonne distance sont un indice plus fort qu'une confiance isolée. |
| Coins compatibles | Distance entre leurs centres comprise entre 2,6 et 5,0 hauteurs de coin (`PAIR_DISTANCE`) | Sur le PNG, les coins sont à 3,7 hauteurs de coin (calculé à partir de `CORNER_TOP_LEFT`). Sur le dataset, les vraies paires vont de 2,9 à 3,9 : la marge couvre la perspective, la rotation et l'imprécision des boîtes prédites. |
| Les deux coins d'une même carte reçoivent deux classes différentes | **Non corrigé** : on obtient deux cartes au lieu d'une | Le cas est mesuré par l'évaluation (« cartes en double »). S'il est fréquent, on pourra regrouper les coins par géométrie, quelle que soit leur classe. |

### L'évaluation ([evaluate.py](evaluate.py))

| Métrique | Définition |
|---|---|
| mAP50, mAP50-95 par coin | Calculées par Ultralytics (`model.val()`) sur le split test. |
| Précision et rappel par carte | Une carte prédite est correcte si la vérité terrain contient une carte de même `class_id`, et qu'au moins un coin prédit a une IoU ≥ 0,5 avec l'un de ses deux coins. On compare à ses deux coins, même un coin en partie caché : le détecter reste une bonne réponse. |
| Cartes en double | Fausses cartes posées sur une vraie carte déjà trouvée : ce sont les deux coins d'une carte qui ont reçu deux classes. |
| Images exactes | Part des images où l'ensemble des cartes prédites est exactement l'ensemble attendu. C'est ce qui compte pour la suite : le LLM ne raisonne bien que sur une liste de cartes juste. |

Contrôle effectué : avec la vérité terrain en entrée, l'évaluation donne une précision, un rappel et une part d'images exactes de 1.

## Résultats

À compléter après l'entraînement complet :

| Modèle | mAP50 coins | Précision cartes | Rappel cartes | Cartes en double | Images exactes |
|---|---|---|---|---|---|
| yolov8n | | | | | |
| yolov8s | | | | | |
