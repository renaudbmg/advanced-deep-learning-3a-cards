"""Génération d'une scène synthétique : quelques cartes posées sur un fond aléatoire.

On annote les deux coins imprimés de chaque carte (rang + enseigne), pas la carte entière.
Comme c'est nous qui posons les cartes, on connaît la boîte exacte de chaque coin et la part
qui reste visible sous les autres cartes : les annotations sont parfaites et gratuites.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from ..deck import DECK
from .decks import DECK_STYLES

# Un coin compte comme visible si au moins 70 % de ses pixels ne sont pas cachés par une
# autre carte. Chaque carte posée garde au moins un coin visible.
MIN_CORNER_VISIBILITY = 0.7


def with_white_card(ink):
    """Dessine la carte blanche sous l'encre, pour les jeux dont le PNG n'a que l'encre."""
    card = Image.new("RGBA", ink.size, (0, 0, 0, 0))
    ImageDraw.Draw(card).rounded_rectangle([0, 0, ink.width - 1, ink.height - 1], radius=12, fill="white", outline=(60, 60, 60))
    card.alpha_composite(ink)
    return card


def load_decks(cards_dir="data/cards"):
    """Liste de (style, {carte: image RGBA}), un élément par jeu de decks.py."""
    decks = []
    for style in DECK_STYLES:
        images = {}
        for card in DECK:
            path = Path(cards_dir) / style.name / card.filename
            if not path.exists():
                raise FileNotFoundError(f"{path} introuvable. Lance d'abord : python -m cards.dataset.assets")
            image = Image.open(path).convert("RGBA")
            images[card] = with_white_card(image) if style.white_fill else image
        decks.append((style, images))
    return decks


def corner_mask(size, style):
    """Masque de la taille de la carte : 0 ailleurs, 1 sur le coin haut-gauche, 2 sur le coin bas-droit."""
    w, h = size
    sx, sy = w / style.card_size[0], h / style.card_size[1]  # au cas où les PNG auraient une autre taille
    x1, y1, x2, y2 = (round(v * s) for v, s in zip(style.corner, (sx, sy, sx, sy)))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    mask[h - y2 : h - y1, w - x2 : w - x1] = 2
    return Image.fromarray(mask)


def random_background(rng, size):
    kind = rng.integers(3)
    if kind == 0:  # couleur unie
        array = np.broadcast_to(rng.integers(0, 256, 3), (size, size, 3)).astype(np.float32)
    elif kind == 1:  # dégradé entre deux couleurs
        t = np.linspace(0, 1, size)[:, None, None]
        array = np.broadcast_to(rng.integers(0, 256, 3) * (1 - t) + rng.integers(0, 256, 3) * t, (size, size, 3))
        if rng.random() < 0.5:
            array = array.transpose(1, 0, 2)
    else:  # taches de couleur floues
        low = rng.integers(0, 256, (4, 4, 3)).astype(np.uint8)
        array = np.asarray(Image.fromarray(low).resize((size, size), Image.BICUBIC), dtype=np.float32)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def augment(image, rng):
    """Retouches de l'image entière, cartes comprises."""
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.7, 1.3))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    # Grain en dernier, comme le bruit d'un capteur de caméra : il couvre aussi les cartes
    array = np.asarray(image, dtype=np.float32) + rng.normal(0, 6, (image.height, image.width, 3))
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def perspective_coefficients(output_points, input_points):
    """Les 8 coefficients qu'attend Image.transform(..., Image.PERSPECTIVE, ...).

    PIL calcule, pour chaque pixel (x, y) de l'image de sortie, la position
    correspondante dans l'image d'entrée :
        ((a x + b y + c) / (g x + h y + 1), (d x + e y + f) / (g x + h y + 1))
    Chaque couple de points (sortie -> entrée) donne deux équations : avec 4 coins,
    on obtient un système linéaire de 8 équations à 8 inconnues.
    """
    matrix, target = [], []
    for (x, y), (u, v) in zip(output_points, input_points):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        target += [u, v]
    return np.linalg.solve(np.array(matrix, dtype=np.float64), np.array(target, dtype=np.float64))


def perspective(sprite, rng, strength, mask=None):
    """Déforme la carte comme si elle était vue en biais.

    Chaque coin est rapproché du centre d'une fraction aléatoire (entre 0 et strength)
    de la largeur et de la hauteur. Les coins ne font que rentrer : la carte déformée
    reste dans son canevas, et ce qui est hors de la carte reste transparent.
    Si un masque de coins est donné, il subit exactement la même déformation.
    """
    w, h = sprite.size
    dx = rng.uniform(0, strength, 4) * w
    dy = rng.uniform(0, strength, 4) * h
    corners = [(0, 0), (w, 0), (w, h), (0, h)]  # haut-gauche, haut-droit, bas-droit, bas-gauche
    moved = [(dx[0], dy[0]), (w - dx[1], dy[1]), (w - dx[2], h - dy[2]), (dx[3], h - dy[3])]
    coefficients = perspective_coefficients(moved, corners)
    sprite = sprite.transform((w, h), Image.PERSPECTIVE, coefficients, Image.BICUBIC)
    if mask is None:
        return sprite
    return sprite, mask.transform((w, h), Image.PERSPECTIVE, coefficients, Image.NEAREST)


def transform_card(sprite, style, rng, height, max_rotation, max_perspective, max_size):
    """Perspective, taille, rotation. Le masque des coins suit chaque étape (en NEAREST, pour garder 0, 1, 2).

    Retourne la carte (RGBA) et son masque de coins, découpés au ras de la carte.
    """
    mask = corner_mask(sprite.size, style)
    # Intensité tirée par carte : de quasiment à plat à nettement penchée
    sprite, mask = perspective(sprite, rng, rng.uniform(0, max_perspective), mask)
    bbox = sprite.getchannel("A").getbbox()
    sprite, mask = sprite.crop(bbox), mask.crop(bbox)
    # Redimensionner après la perspective, qui rétrécit la carte : elle retrouve la hauteur tirée
    size = (round(sprite.width * height / sprite.height), height)
    sprite, mask = sprite.resize(size, Image.LANCZOS), mask.resize(size, Image.NEAREST)
    angle = rng.uniform(-max_rotation, max_rotation)
    sprite = sprite.rotate(angle, resample=Image.BICUBIC, expand=True)
    mask = mask.rotate(angle, resample=Image.NEAREST, expand=True)
    bbox = sprite.getchannel("A").getbbox()  # découpe au ras de la carte déformée
    sprite, mask = sprite.crop(bbox), mask.crop(bbox)
    if max(sprite.size) > max_size:  # garde-fou : la carte doit tenir dans l'image
        scale = max_size / max(sprite.size)
        size = (max(1, int(sprite.width * scale)), max(1, int(sprite.height * scale)))
        sprite, mask = sprite.resize(size, Image.LANCZOS), mask.resize(size, Image.NEAREST)
    return sprite, mask


def place_cards(image, deck, rng, n_cards, base_height, max_rotation, max_perspective, max_tries=50):
    """Colle des cartes au hasard ; elles peuvent se recouvrir.

    Une position est refusée si la nouvelle carte cacherait trop les deux coins d'une carte
    déjà posée. Une carte qui ne trouve pas de place après max_tries essais est abandonnée.

    Retourne la liste des cartes et, pour chacune, ses deux coins :
    [{"box": [x1, y1, x2, y2], "visibility": part des pixels du coin visibles}, ...]
    """
    style, card_images = deck
    size = image.width
    owner = np.full((size, size), -1, dtype=np.int16)  # indice de la carte visible en chaque pixel
    cards, corner_pixels, corner_boxes = [], [], []
    for index in rng.choice(len(DECK), size=n_cards, replace=False):
        card = DECK[index]
        height = round(base_height * rng.uniform(0.9, 1.1))  # ±10 % autour de la taille de l'image
        sprite, mask = transform_card(card_images[card], style, rng, height, max_rotation, max_perspective, size)
        opaque = np.asarray(sprite.getchannel("A")) > 127
        mask = np.asarray(mask)
        k = len(cards)

        for _ in range(max_tries):
            x = int(rng.integers(0, size - sprite.width + 1))
            y = int(rng.integers(0, size - sprite.height + 1))
            new_owner = owner.copy()
            region = new_owner[y : y + sprite.height, x : x + sprite.width]
            region[opaque] = k
            # Chaque carte déjà posée doit garder au moins un coin assez visible
            if all(
                max(visibility(new_owner, pixels, j) for pixels in corner_pixels[j]) >= MIN_CORNER_VISIBILITY
                for j in range(k)
            ):
                break
        else:
            continue  # aucune position acceptable : la carte est abandonnée

        image.paste(sprite, (x, y), sprite)
        owner = new_owner
        cards.append(card)
        pixels, boxes = [], []
        for c in (1, 2):
            ys, xs = np.nonzero((mask == c) & opaque)
            pixels.append((ys + y, xs + x))
            # Boîte du coin entier, même s'il est en partie caché ; x2 et y2 exclusifs
            boxes.append([int(xs.min()) + x, int(ys.min()) + y, int(xs.max()) + x + 1, int(ys.max()) + y + 1])
        corner_pixels.append(pixels)
        corner_boxes.append(boxes)

    corners = [
        [{"box": box, "visibility": round(visibility(owner, pixels, j), 3)} for box, pixels in zip(boxes, corner_pixels[j])]
        for j, boxes in enumerate(corner_boxes)
    ]
    return cards, corners


def visibility(owner, pixels, card_index):
    """Part des pixels d'un coin où la carte est encore celle du dessus."""
    ys, xs = pixels
    return float(np.mean(owner[ys, xs] == card_index))


# Probabilité de chaque nombre de cartes visé : 0, 1, 2, 3, 4, 5.
N_CARDS_PROBS = (0.05, 0.20, 0.25, 0.25, 0.15, 0.10)

# Hauteur des cartes, en pixels, pour une image de 384 px. On tire une hauteur par image,
# comme une distance à la caméra : toutes les cartes d'une image ont à peu près la même
# taille. La hauteur maximale baisse avec le nombre de cartes, pour qu'elles tiennent dans
# l'image. Le minimum garde des coins d'environ 10 × 19 px.
MIN_CARD_HEIGHT = 75
MAX_CARD_HEIGHT = {0: 300, 1: 300, 2: 225, 3: 180, 4: 160, 5: 140}


def make_scene(
    decks,
    rng,
    image_size=384,
    n_cards_probs=N_CARDS_PROBS,
    max_rotation=20,
    max_perspective=0.15,
):
    """Retourne (image RGB, liste de cartes, liste des deux coins de chaque carte, nom du jeu).

    Le nombre de cartes posées peut être inférieur au nombre visé : une carte qui ne
    trouve pas de place acceptable est abandonnée.
    """
    image = random_background(rng, image_size)
    n = int(rng.choice(len(n_cards_probs), p=n_cards_probs))
    scale = image_size / 384  # les hauteurs sont données pour 384 px
    base_height = rng.uniform(MIN_CARD_HEIGHT, MAX_CARD_HEIGHT[n]) * scale
    deck = decks[int(rng.integers(len(decks)))]
    cards, corners = place_cards(image, deck, rng, n, base_height, max_rotation, max_perspective)
    return augment(image, rng), cards, corners, deck[0].name
