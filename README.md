# advanced-deep-learning-3a

Projet de Deep Learning avancé : question-réponse multimodal (image + texte) sur des cartes à jouer.

On montre une image de cartes et on pose une question en langage naturel, par exemple « les cartes rouges ». Le système renvoie l'image avec les boîtes des seules cartes qui répondent à la question : un YOLO pré-entraîné, fine-tuné sur notre dataset, détecte les cartes, puis un LLM répond à la question à partir de la liste des cartes détectées et de leurs positions.

## Équipe

- Renaud Baumgarten
- Romain Chenot
- Mathis Isaac

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Structure

- `cards/` : le code du projet, un sous-dossier par étape, chacun avec un README qui explique ses décisions.
  - [cards/deck.py](cards/deck.py) : les 52 cartes et leurs attributs, partagés par toutes les étapes.
  - [cards/dataset/](cards/dataset/) : génération du dataset synthétique de cartes à jouer ([README](cards/dataset/README.md)).
- `notebooks/` : notebooks d'exploration communs.
- `docs/` : rapport, supports de soutenance, liens utiles.

Le dossier `data/` (cartes téléchargées, dataset généré) n'est pas versionné.

## Démarrage rapide

Depuis la racine du repo :

```bash
python -m cards.dataset.assets                    # télécharge les 52 cartes dans data/cards/
python -m cards.dataset.generate_dataset --yolo   # génère le dataset dans data/synthetic/
```
