"""Les jeux de cartes (designs) utilisés pour générer le dataset.

Plusieurs designs libres de droits obligent le modèle à apprendre ce qui est commun à tous les jeux (le rang
et l'enseigne dans le coin), pas une police ou une mise en page particulière.

Tous viennent de Wikimedia Commons, en PNG de 250 px de large.
"""

from dataclasses import dataclass
from typing import Callable

from ..deck import Card

RANK_SHORT = {"ace": "A", "jack": "J", "queen": "Q", "king": "K"}


@dataclass(frozen=True)
class DeckStyle:
    name: str  # dossier dans data/cards/
    title: Callable[[Card], str]  # titre du fichier SVG sur Wikimedia Commons
    source: str
    license: str
    card_size: tuple  # taille des PNG téléchargés, en pixels
    corner: tuple  # coin haut-gauche (rang + enseigne), [x1, y1, x2, y2[ en pixels ; le bas-droit est son symétrique
    white_fill: bool = False  # le PNG ne contient que l'encre : il faut dessiner la carte blanche dessous


def byron_knoll_title(card):
    rank = card.rank if card.rank.isdigit() else card.rank.capitalize()
    variant = "2" if card.is_face else ""  # "King of hearts2.svg" : la version illustrée des figures
    return f"File:{rank} of {card.suit}{variant}.svg"


def english_pattern_title(card):
    return f"File:English pattern {card.rank} of {card.suit}.svg"


def cards_1_title(card):
    rank = RANK_SHORT.get(card.rank, card.rank)
    return f"File:Cards-{rank}-{card.suit[:-1].capitalize()}.svg"  # "Cards-K-Club.svg"


def cards_6_title(card):
    rank = RANK_SHORT.get(card.rank, card.rank)
    return f"File:{rank}{card.suit[0].upper()}.svg"  # "KC.svg"


DECK_STYLES = [
    DeckStyle(
        "byron_knoll",
        byron_knoll_title,
        "https://commons.wikimedia.org/wiki/Category:Playing_cards_set_by_Byron_Knoll",
        "domaine public",
        (250, 364),
        (2, 2, 45, 93),
    ),
    DeckStyle(
        "english_pattern",
        english_pattern_title,
        "https://commons.wikimedia.org/wiki/Category:SVG_English_pattern_playing_cards",
        "CC0",
        (250, 375),
        (5, 19, 34, 97),
    ),
    DeckStyle(
        "cards_1",
        cards_1_title,
        "https://commons.wikimedia.org/wiki/Category:SVG_playing_cards_1",
        "domaine public",
        (250, 350),
        (3, 12, 45, 99),
        white_fill=True,
    ),
    DeckStyle(
        "cards_6",
        cards_6_title,
        "https://commons.wikimedia.org/wiki/Category:SVG_playing_cards_6",
        "CC0",
        (250, 350),
        (9, 10, 35, 79),
    ),
]

# Les rectangles de coin ont été mesurés sur les 40 cartes non figures de chaque jeu (zone d'encre
# dans le coin haut-gauche, liseré exclu, 2 px de marge), puis vérifiés à l'œil sur des figures.
