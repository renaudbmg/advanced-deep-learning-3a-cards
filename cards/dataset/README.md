# Génération du dataset

Responsable : Renaud Baumgarten

Ce dossier génère un dataset synthétique d'images de cartes à jouer, avec le **nom de chaque carte et les boîtes de ses deux coins** (rang + enseigne).

## Utilisation

Depuis la racine du repo :

```bash
python -m cards.dataset.assets              # 1. télécharge les 52 cartes dans data/cards/ (une seule fois)
python -m cards.dataset.generate_dataset    # 2. génère le dataset dans data/synthetic/
```

Options de `generate_dataset` :

| Option | Défaut | Rôle |
|---|---|---|
| `--n-train`, `--n-val`, `--n-test` | 20 000, 2 000, 2 000 | Nombre d'images par split |
| `--out-dir` | `data/synthetic` | Dossier de sortie |
| `--cards-dir` | `data/cards` | Dossier des images de cartes |
| `--seed` | 0 | Graine du split train ; val et test utilisent `seed + 1` et `seed + 2` |
| `--yolo` | désactivé | Exporte aussi les trois splits au format YOLO, pour le fine-tuning |

Le notebook [visualisation.ipynb](visualisation.ipynb) affiche des exemples d'images avec leurs boîtes, et des statistiques sur le dataset.

## Les fichiers

| Fichier | Rôle |
|---|---|
| [../deck.py](../deck.py) | (partagé avec tout le package) Les 52 cartes et leurs attributs : rang, enseigne, couleur, figure ou non, nom, `class_id`. |
| [assets.py](assets.py) | Télécharge les images des cartes depuis Wikimedia Commons. |
| [scene.py](scene.py) | Compose une image : fond, placement des cartes, retouches. Renvoie l'image, les cartes et les boîtes de leurs coins. |
| [generate_dataset.py](generate_dataset.py) | Répète la composition pour chaque image, écrit les images et `annotations.json`, et l'export YOLO. |

## Format des données produites

C'est ce que lit l'entraînement.

```
data/synthetic/{train,val,test}/images/000000.jpg
data/synthetic/{train,val,test}/annotations.json
data/synthetic/yolo/{train,val,test}/{images,labels}/   (avec --yolo)
data/synthetic/yolo/data.yaml
```

```json
{
  "image_size": 384,
  "min_corner_visibility": 0.7,
  "items": [{
    "image": "images/000000.jpg",
    "cards": [
      {"name": "7 of clubs", "class_id": 31, "corners": [
        {"box": [88, 88, 128, 147], "visibility": 0.0, "visible": false},
        {"box": [260, 208, 305, 271], "visibility": 1.0, "visible": true}
      ]},
      {"name": "4 of spades", "class_id": 41, "corners": [
        {"box": [62, 57, 98, 116], "visibility": 1.0, "visible": true},
        {"box": [215, 199, 252, 257], "visibility": 1.0, "visible": true}
      ]}
    ]
  }]
}
```

- `corners` : les deux coins imprimés de la carte, haut-gauche puis bas-droit **de la carte** (avant rotation).
- `box` : `[x1, y1, x2, y2]` en pixels, coins haut-gauche et bas-droit de la boîte. `x2` et `y2` sont exclusifs : la largeur vaut `x2 − x1`. La boîte couvre le coin entier, même s'il est en partie caché.
- `visibility` : part des pixels du coin qui ne sont pas cachés par une autre carte, entre 0 et 1.
- `visible` : `visibility ≥ min_corner_visibility` (0,7). Chaque carte a au moins un coin visible.
- `class_id` : numéro de la carte de 0 à 51, égal à `enseigne × 13 + rang` (enseignes : hearts, diamonds, clubs, spades ; rangs : 2 à 10, jack, queen, king, ace).
- `cards` peut être vide : c'est une image sans carte.
- Export YOLO : un fichier `.txt` par image, une ligne `class_id cx cy w h` par **coin visible** (valeurs entre 0 et 1), et un fichier vide pour une image sans carte. Les coins trop cachés ne sont pas exportés. `data.yaml` donne les chemins des trois splits et les 52 noms de classes : c'est le fichier à passer à Ultralytics.

**Une carte = un `class_id`.** Les cartes sont tirées sans remise : une carte apparaît au plus une fois par image. Deux coins détectés avec le même `class_id` appartiennent donc forcément à la même carte, et c'est ainsi qu'on regroupe les détections par carte.

## Décisions

### Annoter les coins plutôt que la carte entière

Comme le dataset Kaggle utilisé par [TeogopK/Playing-Cards-Object-Detection](https://github.com/TeogopK/Playing-Cards-Object-Detection) (généré par [geaxgx](https://github.com/geaxgx/playing-card-detection)), on annote les **deux coins imprimés** de chaque carte, pas la carte entière.

| Raison | Détail |
|---|---|
| Le coin reste visible quand les cartes se recouvrent | Dans une main en éventail ou une pile, on ne voit souvent que les coins. On peut donc laisser les cartes se recouvrir, comme sur une vraie table. |
| Tout ce qui identifie la carte est dans le coin | Le rang et l'enseigne. Le détecteur se concentre sur la zone qui compte. |
| Les boîtes restent serrées malgré la rotation | Une boîte droite autour d'une carte tournée de 20° contient beaucoup de fond ; autour d'un petit coin, très peu. |
| Deux coins par carte | Si un coin est caché, l'autre suffit à trouver la carte. |

Le prix à payer : les coins sont de petits objets. C'est pourquoi l'image passe à 384 px et la hauteur minimale des cartes à 75 px (voir plus bas).

### Les images des cartes

| Décision | Pourquoi |
|---|---|
| Jeu de **Byron Knoll**, dans le domaine public | Libre de droits, complet (52 cartes) et au design standard. |
| Téléchargement via l'**API Wikimedia**, qui fournit une version PNG de 250 px de large |
| Zone du coin : `[2, 2, 45, 93[` sur le PNG de 250×364 (`CORNER_TOP_LEFT`), et le même rectangle tourné de 180° pour le coin bas-droit | Mesurée sur les 52 cartes, qui ont toutes la même mise en page. Le rectangle contient le rang et l'enseigne, y compris le « 10 » plus large, sans déborder sur les motifs du centre. |

### La composition d'une image ([scene.py](scene.py))

| Décision | Valeur | Pourquoi |
|---|---|---|
| Taille de l'image | 384×384 | Les coins sont petits : à 256 px, le coin d'une petite carte ne faisait que 8×12 px. 384 px reste entraînable sur un portable. |
| Nombre de cartes visé | 0 : 5 %, 1 : 20 %, 2 : 25 %, 3 : 25 %, 4 : 15 %, 5 : 10 % (`N_CARDS_PROBS`) | Les **images vides** apprennent au détecteur à ne rien voir quand il n'y a rien, au lieu d'inventer une carte dans le fond. Les **images à une carte** imitent le cas typique d'une démo : on montre une carte à la webcam. |
| Tirage des cartes | Sans remise | Une carte apparaît au plus une fois par image, comme avec un vrai jeu. |
| Hauteur des cartes | Une hauteur par image, entre 75 px et un maximum qui dépend du nombre de cartes (300 px pour 0 ou 1 carte, 225 pour 2, 180 pour 3, 160 pour 4, 140 pour 5), puis ±10 % par carte (`MIN_CARD_HEIGHT`, `MAX_CARD_HEIGHT`) | À la webcam, une carte peut être tenue près de l'objectif ou posée loin. Tirer la taille par image imite une distance à la caméra : les cartes posées sur une même table ont à peu près la même taille. Le maximum baisse avec le nombre de cartes pour qu'elles tiennent toutes dans l'image. Le minimum garde des coins d'environ 10×19 px. |
| Perspective | Intensité tirée par carte entre 0 et 15 % (`max_perspective`) ; chaque coin est rapproché du centre d'une fraction aléatoire de cette intensité | Imite une carte vue en biais, qui devient un trapèze. L'intensité par carte garde aussi des cartes presque à plat (environ 20 %), comme une carte tenue face à la caméra. |
| Rotation | −20° à +20° | Imite des cartes posées à la main, sans les rendre méconnaissables. |
| Position | Tirée au hasard, carte toujours **entièrement** dans l'image | Simple, et chaque boîte correspond à une carte complète. |
| Recouvrement | Autorisé, tant que **chaque carte garde au moins un coin visible à 70 %** (`MIN_CORNER_VISIBILITY`) | Les cartes peuvent se chevaucher comme sur une vraie table. La seule chose qui compte pour le détecteur est de pouvoir lire au moins un coin de chaque carte annotée. |
| Essais de placement | 50 par carte | Au-delà, la carte est abandonnée et l'image en a une de moins. |

**La règle de visibilité** se calcule au pixel près. On tient une carte `owner` de l'image qui indique, pour chaque pixel, quelle carte est au-dessus. Quand on essaie de poser une nouvelle carte, on met à jour une copie de `owner`, puis on vérifie que chaque carte déjà posée a encore un coin dont au moins 70 % des pixels lui appartiennent. Sinon, la position est refusée. La nouvelle carte est posée par-dessus, donc elle est entièrement visible. Le seuil de 70 % laisse un coin lisible : un coin caché à moitié peut perdre son enseigne ou une partie du rang.

### Le calcul des boîtes des coins

Les boîtes ne sont pas déduites de l'image finale : elles viennent de la façon dont on colle chaque carte (`transform_card()` et `place_cards()`).

1. On crée un **masque des coins** de la taille du PNG (`corner_mask()`) : 0 ailleurs, 1 sur le coin haut-gauche, 2 sur le coin bas-droit.
2. La carte est déformée en perspective (`perspective()`, avec `Image.transform` de PIL et les coefficients calculés par `perspective_coefficients()`), recadrée, redimensionnée à la hauteur tirée, puis tournée avec `expand=True`. **Le masque subit exactement les mêmes opérations**, avec un rééchantillonnage `NEAREST` pour garder les valeurs 0, 1 et 2. Le redimensionnement vient après la perspective, qui rétrécit la carte : la hauteur tirée est ainsi respectée.
3. On découpe la carte et le masque au ras des pixels non transparents de la carte, puis on tire la position `(x, y)` de la carte dans l'image.
4. La boîte d'un coin est la plus petite boîte droite qui contient ses pixels dans le masque, décalée de `(x, y)`.
5. On colle la carte avec sa transparence comme masque : les coins arrondis laissent voir le fond.

Les boîtes sont donc **exactes au pixel près**, sans annotation manuelle. Elles sont alignées sur les axes, comme en détection classique. La `visibility` de chaque coin est calculée à la fin, quand toutes les cartes sont posées, avec la carte `owner`.

### Le fond et les retouches

| Décision | Valeur | Pourquoi |
|---|---|---|
| Type de fond | Couleur unie, dégradé (vertical ou horizontal) ou taches de couleur floues, 1 chance sur 3 chacun | Le détecteur ne peut pas apprendre un raccourci lié à un fond particulier. |
| Luminosité, contraste | Facteur entre 0,7 et 1,3, sur l'image entière | Imite des éclairages différents. |
| Flou gaussien | 30 % des images, rayon 0,3 à 1,2 px | Imite une mise au point imparfaite. |
| Grain | Bruit gaussien, écart-type 6 sur 255, ajouté en dernier sur l'image entière, cartes comprises | Imite le bruit d'un capteur. Il était d'abord appliqué au fond seulement : les cartes, parfaitement nettes, se distinguaient du fond par ce seul détail, un raccourci qu'un détecteur aurait pu apprendre. |
| Format | JPEG qualité 92 | Fichiers légers, artefacts de compression faibles et proches d'une vraie caméra. |

### La production du dataset ([generate_dataset.py](generate_dataset.py))

| Décision | Pourquoi |
|---|---|
| Génération **sur disque**, une fois pour toutes | On peut ouvrir les images pour vérifier, le dataset est le même pour tout le groupe, et la génération ne ralentit pas l'entraînement. |
| **Graine fixe** par split | Relancer la commande redonne exactement les mêmes images, et train, val et test sont différents. |