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
| `train` | `--model` (défaut `yolov8s.pt`), `--epochs` (50), `--patience` (10), `--batch` (32), `--device` (`mps`), `--data`, `--name` |
| `predict` | `--weights`, `--conf` (0,25), `--imgsz` (384), `--out` (défaut `runs/predict/<nom>.jpg`) |
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
| [real_dataset.py](real_dataset.py) | Télécharge les vraies photos de test et convertit leurs étiquettes. |
| [evaluate_real.py](evaluate_real.py) | Métriques sur les vraies photos, et images annotées des erreurs. |

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
| Coins compatibles | Distance entre leurs centres comprise entre 2,5 et 6,5 longueurs de coin (`PAIR_DISTANCE`) ; la longueur d'un coin est le grand côté de sa boîte | Selon le jeu, les coins sont à 3,6 à 4,8 longueurs de coin (calculé à partir de `DeckStyle.corner`). Le grand côté de la boîte ne dépend pas de la rotation : avec la hauteur, une carte couchée à 90° aurait vu ses deux coins séparés. Vérifié sur le dataset : toutes les vraies paires de coins sont appariées. |
| Les deux coins d'une même carte reçoivent deux classes différentes | **Non corrigé** : on obtient deux cartes au lieu d'une | Le cas est mesuré par l'évaluation (« cartes en double »). S'il est fréquent, on pourra regrouper les coins par géométrie, quelle que soit leur classe. |

### L'évaluation ([evaluate.py](evaluate.py))

| Métrique | Définition |
|---|---|
| mAP50, mAP50-95 par coin | Calculées par Ultralytics (`model.val()`) sur le split test. |
| Précision et rappel par carte | Une carte prédite est correcte si la vérité terrain contient une carte de même `class_id`, et qu'au moins un coin prédit a une IoU ≥ 0,5 avec l'un de ses deux coins. On compare à ses deux coins, même un coin en partie caché : le détecter reste une bonne réponse. |
| Cartes en double | Fausses cartes posées sur une vraie carte déjà trouvée : ce sont les deux coins d'une carte qui ont reçu deux classes. |
| Images exactes | Part des images où l'ensemble des cartes prédites est exactement l'ensemble attendu. C'est ce qui compte pour la suite : le LLM ne raisonne bien que sur une liste de cartes juste. |

Contrôle effectué : avec la vérité terrain en entrée, l'évaluation donne une précision, un rappel et une part d'images exactes de 1.

## Évaluation sur photos réelles

Le split test synthétique vient du même générateur que l'entraînement : mêmes 52 images de cartes, mêmes fonds simples. Il mesure si la tâche est apprise, pas si le modèle tient sur de vraies photos. On ajoute donc un test **hors distribution** : le modèle n'est entraîné que sur le synthétique, et testé sur de vraies photos qui ne servent jamais à l'entraîner.

```bash
python -m cards.detection.real_dataset --split test          # 193 photos, pour le chiffre final
python -m cards.detection.real_dataset --split validation    # 128 photos, pour régler --conf et --imgsz
python -m cards.detection.evaluate_real --weights checkpoints/yolov8s_ft35.pt --data-dir data/real/poker-cards/validation
python -m cards.detection.evaluate_real --weights checkpoints/yolov8s_ft35.pt
```

Les images des photos avec une erreur (prédictions en vert et jaune, vraies boîtes en bleu, cartes manquées et fantômes écrites en rouge) et le détail par photo (`results.json`) sont écrits dans `runs/real_eval/<poids>/<split>_<imgsz>/`.

### Le dataset

[Francesco/poker-cards-cxcvz](https://huggingface.co/datasets/Francesco/poker-cards-cxcvz) sur Hugging Face, issu de Roboflow 100 : de vraies photos 640×640, 4 à 5 cartes par photo, posées sur des tissus, des canapés, des tables. Licence CC BY 4.0 d'après les miroirs du dataset (le dépôt indique seulement « cc ») : à citer, et à vérifier sur la page Roboflow d'origine.

| Décision | Pourquoi |
|---|---|
| Split `test` pour le chiffre final, `validation` pour les réglages, `train` jamais utilisé | Le test reste un juge neutre : aucun choix n'est fait en le regardant. |
| Comparaison par les **noms** des cartes, pas par les boîtes | Leurs boîtes couvrent la carte entière, les nôtres couvrent les coins : une IoU entre les deux serait basse même pour une détection parfaite. |
| **Correction des numéros de classe** (`true_category_name()` dans [real_dataset.py](real_dataset.py)) | Dans le fichier COCO d'origine, la liste de noms est décalée pour les classes 1 à 21 : la classe « 59 » est le 10 de cœur, « 10 Diamonds » est le 10 de pique, etc. Le décalage a été repéré en dessinant les étiquettes sur les photos, puis vérifié classe par classe en recadrant les boîtes. Après correction, les 52 cartes sont présentes dans chaque split, et aucun objet n'est écarté. |

### Les métriques

| Métrique | Définition |
|---|---|
| Précision, rappel par carte | Une carte prédite est juste si la photo contient une carte de même nom. |
| Cartes fantômes, manquées | Cartes prédites absentes de la photo ; cartes de la photo non trouvées. |
| Images exactes | Part des photos où l'ensemble des cartes prédites est exactement l'ensemble attendu. |
| Cartes bien localisées | Parmi les cartes bien identifiées, part de celles dont un coin détecté a son centre dans la vraie boîte de la carte. |
| Relations spatiales | Pour chaque paire de cartes bien identifiées, « à gauche de » et « au-dessus de » sont-ils les mêmes entre les centres prédits et les vrais centres ? Les paires presque alignées (écart réel de moins d'un quart de la taille moyenne des deux cartes) ne sont pas comptées. C'est ce dont le LLM aura besoin. |
| IoU des cartes à deux coins (indicatif) | Quand les deux coins sont détectés, leur union recouvre à peu près la carte : IoU avec la vraie boîte. Approximatif pour les cartes inclinées. |

Contrôle effectué : avec la vérité en entrée, toutes les métriques valent 1 ; en retirant une carte et en ajoutant une fantôme, les compteurs sont exactement ceux attendus.

### Réglages choisis sur validation

| Réglage | Images exactes sur validation | Choix |
|---|---|---|
| `--imgsz 384` / `640` | 0,125 / 0,117 (rappel 0,551 / 0,564) | **384**, la taille de l'entraînement : l'écart est dans le bruit pour 128 photos. |
| `--conf 0,1` / `0,25` / `0,5` | 0,102 / 0,125 / 0,109 | **0,25** : en dessous, trop de fantômes ; au-dessus, trop de cartes manquées. |

## Résultats

Modèle `yolov8s` fine-tuné 35 epochs sur Colab (`checkpoints/yolov8s_ft35.pt`), `--conf 0,25`, `--imgsz 384`.

| Métrique | Test synthétique (2 000 images) | Test réel (193 photos) |
|---|---|---|
| mAP50 / mAP50-95 par coin | 0,995 / 0,909 | non mesurable (pas de boîtes de coin) |
| Précision par carte | 1,000 | **0,815** |
| Rappel par carte | 0,999 | **0,557** |
| Cartes fantômes | 0 | 106 |
| Images exactes | 0,999 | **0,114** |
| Cartes bien localisées | | 0,949 |
| Relations gauche/droite, haut/bas | | 0,952 / 0,961 |
| IoU des cartes à deux coins (moyenne, part ≥ 0,5) | | 0,597 ; 0,729 |

**Lecture :**
- **L'écart entre synthétique et réel est grand.** Sur le synthétique, la tâche est résolue ; sur de vraies photos, le modèle ne trouve que 56 % des cartes, et seule une photo sur neuf est entièrement juste.
- **Quand une carte est trouvée, elle est bien placée** : 95 % des coins sont sur la bonne carte, et les relations gauche/droite et haut/bas sont justes à 95-96 %. Le problème est surtout la **reconnaissance**, pas la position.
- Le rappel varie selon le rang : 0,30 pour les 5, 0,44 à 0,47 pour les 3, 4 et 10, jusqu'à 0,77 pour les 6.
- Une confusion visible sur les images d'erreurs : le coin d'un **6 vu à l'envers est lu comme un 9**.
- Causes probables, à vérifier : le jeu des photos n'a pas la même police ni la même mise en page que le jeu de Byron Knoll du générateur, les cartes sont plus inclinées que les ±20° du générateur, et les fonds sont texturés. Une vraie photo d'une quinte flush à pique ne donnait que 3 cartes sur 5, et les 5 une fois l'image redressée de 30°.

Ces résultats ont conduit à une **version 2 du générateur** : 4 jeux de cartes, rotations à 360°, fonds texturés, leurres, éclairage et compression (détail dans le [README du dataset](../dataset/README.md#pourquoi-le-générateur-a-été-durci-version-2)).

### Version 2 du générateur

Modèle `yolov8s` fine-tuné sur la version 2 avec le [notebook Colab](colab_training.ipynb). L'entraînement a été coupé par la limite de GPU de Colab après **29 epochs sur 50** : on garde les poids de l'epoch 29 (la meilleure du CSV, `mAP50-95` de validation synthétique de 0,8685, qui montait encore de 0,001 par epoch). Les poids sont sur Google Drive (`adl3a_runs/yolov8s_checkpoints/best.pt`) et en local dans `checkpoints/yolov8s_v2_ep29_best.pt`, hors git.

**Réglage du seuil de confiance, sur le split `validation` réel** (128 photos, `--imgsz 384`) :

| `--conf` | Précision | Rappel | Fantômes | Manquées | Images exactes |
|---|---|---|---|---|---|
| 0,25 | 0,832 | 0,975 | 109 | 14 | 0,438 |
| 0,40 | 0,870 | 0,975 | 81 | 14 | 0,531 |
| 0,55 | 0,910 | 0,971 | 53 | 16 | 0,625 |
| 0,70 | 0,953 | 0,946 | 26 | 30 | 0,664 |
| **0,80** | **0,986** | 0,903 | 7 | 54 | **0,688** |

`--imgsz 640` n'améliore rien (0,492 images exactes à `conf 0,25`, 0,648 à 0,7, 0,641 à 0,8) : on garde 384. Les fausses cartes ont une confiance basse, donc un seuil plus haut les enlève presque toutes au prix de quelques cartes manquées. Entre 0,70 et 0,80, l'écart est de 3 images sur 128, dans le bruit ; on retient **0,80**, qui maximise les images exactes (le critère fixé d'avance).

**Résultats finaux, `--conf 0,8` et `--imgsz 384` pour toutes les colonnes** (le test réel n'a été lancé qu'une fois, après ces réglages) :

| Métrique | v1 : test synthétique v1 | v1 : test réel | v2 : test synthétique v2 | v2 : test réel |
|---|---|---|---|---|
| Précision par carte | 1,000 | 0,815 | 1,000 | **0,988** |
| Rappel par carte | 0,999 | 0,557 | 0,971 | **0,871** |
| Cartes fantômes | 0 | 106 | 1 | 9 |
| Images exactes | 0,999 | 0,114 | 0,915 | **0,622** |
| mAP50 / mAP50-95 par coin | 0,995 / 0,909 | non mesurable | 0,994 / 0,866 | non mesurable |
| Cartes bien localisées | | 0,949 | | 1,000 |
| Relations gauche/droite, haut/bas | | 0,952 / 0,961 | | 0,991 / 0,986 |
| IoU des cartes à deux coins (moyenne, part ≥ 0,5) | | 0,597 ; 0,729 | | 0,517 ; 0,572 |

Les colonnes v1 sont celles du tableau précédent, mesurées à `conf 0,25` ; les colonnes v2 à `conf 0,8`. À `conf 0,25`, le v2 donne sur la validation réelle 0,832 de précision, 0,975 de rappel et 0,438 d'images exactes (voir le tableau de réglage ci-dessus). Le test synthétique v2 est régénéré en local avec la même graine qu'à l'entraînement (`--n-test 2000`, seed fixe par split).

**Lecture :**
- **Le gain sur de vraies photos est très net** : les images exactes passent de 11 % à 62 %, le rappel de 0,56 à 0,87 et la précision de 0,82 à 0,99. Le modèle ne se trompe presque plus de carte (9 fausses cartes sur 739 annoncées) ; ce qui reste, ce sont des cartes manquées (108 sur 838), surtout des 5 et des 6 (rappel de 0,74 et 0,79 sur validation, au moins 0,86 pour les autres rangs).
- **La position est quasi parfaite** : 100 % des cartes trouvées sont bien placées, et l'ordre gauche/droite et haut/bas est juste à 99 %. C'est ce dont le LLM aura besoin.
- **La tâche synthétique est plus difficile que celle du v1** (0,915 d'images exactes contre 0,999), comme prévu : les images contiennent des fonds texturés, des rotations à 360° et plus de cartes.