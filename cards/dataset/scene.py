"""Génération d'une scène synthétique : quelques cartes posées sur un fond aléatoire.

On annote les deux coins imprimés de chaque carte (rang + enseigne), pas la carte entière.
Comme c'est nous qui posons les cartes, on connaît la boîte exacte de chaque coin et la part
qui reste visible sous les autres cartes : les annotations sont parfaites et gratuites.

Les variations (designs, rotations, fonds, éclairage) imitent ce qu'on voit sur de vraies photos.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from ..deck import DECK
from .decks import DECK_STYLES

# Un coin compte comme visible si au moins 70 % de ses pixels ne sont pas cachés par une
# autre carte. Chaque carte posée garde au moins un coin visible.
MIN_CORNER_VISIBILITY = 0.7

# Rotation : une carte sur deux reste proche de la verticale (±30°), comme posée ou tenue face à
# la caméra ; l'autre prend n'importe quelle orientation, comme sur une table vue de dessus.
SMALL_ROTATION = 30
FULL_ROTATION_PROB = 0.5


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


# --- Fonds -------------------------------------------------------------------------------------


def two_colors(rng, t):
    """Mélange deux couleurs au hasard selon t (entre 0 et 1, une valeur par pixel)."""
    c1, c2 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    return t[..., None] * c1 + (1 - t[..., None]) * c2


def noise_texture(rng, size):
    """Bruit à plusieurs échelles : tissu, moquette, crépi, bois flou."""
    total = np.zeros((size, size), np.float32)
    for cells, weight in ((4, 1.0), (16, 0.5), (64, 0.3), (size, 0.25)):
        low = rng.random((cells, cells)).astype(np.float32)
        total += weight * np.asarray(Image.fromarray(low).resize((size, size), Image.BICUBIC))
    total = (total - total.min()) / (np.ptp(total) + 1e-6)
    return two_colors(rng, total)


def stripes(rng, size):
    """Rayures, ou carreaux quand on croise deux séries de rayures."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    def series(angle):
        period = rng.uniform(6, 60)
        wave = np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * 2 * np.pi / period)
        return (wave > rng.uniform(-0.5, 0.5)).astype(np.float32)

    angle = rng.uniform(0, np.pi)
    t = series(angle)
    if rng.random() < 0.5:
        t = (t + series(angle + np.pi / 2)) / 2
    return np.asarray(Image.fromarray(two_colors(rng, t).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7)))


def clutter(rng, size):
    """Fouillis de formes : objets sur une table, motifs de tissu, coussins."""
    image = Image.new("RGB", (size, size), tuple(int(v) for v in rng.integers(0, 256, 3)))
    draw = ImageDraw.Draw(image)
    for _ in range(int(rng.integers(10, 40))):
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        x, y = (int(v) for v in rng.integers(-size // 4, size, 2))
        w, h = (int(v) for v in rng.integers(5, size // 2, 2))
        kind = rng.integers(3)
        if kind == 0:
            draw.ellipse([x, y, x + w, y + h], fill=color)
        elif kind == 1:
            draw.rectangle([x, y, x + w, y + h], fill=color)
        else:
            draw.line([x, y, x + w, y + h], fill=color, width=int(rng.integers(1, 8)))
    return np.asarray(image.filter(ImageFilter.GaussianBlur(rng.uniform(0, 2))))


# Probabilité de chaque type de fond : uni, dégradé, taches floues, bruit, rayures, fouillis.
BACKGROUND_PROBS = (0.1, 0.1, 0.15, 0.25, 0.15, 0.25)


def random_background(rng, size):
    kind = rng.choice(6, p=BACKGROUND_PROBS)
    if kind == 0:  # couleur unie
        array = np.broadcast_to(rng.integers(0, 256, 3), (size, size, 3)).astype(np.float32)
    elif kind == 1:  # dégradé entre deux couleurs
        t = np.broadcast_to(np.linspace(0, 1, size)[:, None], (size, size))
        array = two_colors(rng, t.T if rng.random() < 0.5 else t)
    elif kind == 2:  # taches de couleur floues
        low = rng.integers(0, 256, (4, 4, 3)).astype(np.uint8)
        array = np.asarray(Image.fromarray(low).resize((size, size), Image.BICUBIC), dtype=np.float32)
    elif kind == 3:
        array = noise_texture(rng, size)
    elif kind == 4:
        array = stripes(rng, size)
    else:
        array = clutter(rng, size)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def add_decoys(image, rng, base_height, max_decoys=2):
    """Objets en forme de carte, sans rang ni enseigne (dos de carte, papiers) : non annotés.

    Ils apprennent au modèle qu'un rectangle blanc ou coloré n'est pas une carte : sur de vraies
    photos, une partie des fausses détections tombait sur ce genre d'objets.
    """
    for _ in range(int(rng.integers(1, max_decoys + 1))):
        h = max(10, int(base_height * rng.uniform(0.7, 1.3)))
        w = int(h * rng.uniform(0.6, 0.8))
        sprite = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sprite)
        white = rng.random() < 0.5
        fill = (245, 245, 245) if white else tuple(int(v) for v in rng.integers(0, 256, 3))
        draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=max(2, w // 12), fill=fill, outline=(60, 60, 60))
        if not white:  # dos de carte : motif dans un cadre
            m = max(2, w // 10)
            inner = Image.fromarray(stripes(rng, max(w, h))).crop((0, 0, w - 2 * m, h - 2 * m))
            sprite.paste(inner, (m, m))
        sprite = sprite.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True)
        x = int(rng.integers(-sprite.width // 2, image.width - sprite.width // 2))
        y = int(rng.integers(-sprite.height // 2, image.height - sprite.height // 2))
        image.paste(sprite, (x, y), sprite)


# --- Retouches ---------------------------------------------------------------------------------


def augment(image, rng):
    """Retouches de l'image entière, cartes comprises : éclairage, couleur, netteté, compression."""
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.6, 1.4))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.7, 1.3))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.7, 1.3))
    array = np.asarray(image, dtype=np.float32)
    array = array * rng.uniform(0.85, 1.15, 3)  # dominante de couleur : lampe chaude ou froide
    size = image.width
    if rng.random() < 0.5:  # lumière inégale : un côté plus sombre que l'autre
        low = rng.uniform(0.6, 1.15, (3, 3)).astype(np.float32)
        array = array * np.asarray(Image.fromarray(low).resize((size, size), Image.BICUBIC))[..., None]
    if rng.random() < 0.2:  # reflet : une tache claire, comme une lampe sur une carte glacée
        glare = Image.new("L", (size, size), 0)
        cx, cy = rng.integers(0, size, 2)
        r = rng.uniform(0.05, 0.25) * size
        ImageDraw.Draw(glare).ellipse([cx - r, cy - r * 0.6, cx + r, cy + r * 0.6], fill=int(rng.uniform(80, 200)))
        array = array + np.asarray(glare.filter(ImageFilter.GaussianBlur(r / 3)), dtype=np.float32)[..., None]
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    if rng.random() < 0.35:  # mise au point imparfaite
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.8)))
    if rng.random() < 0.25:  # basse résolution, comme une webcam : réduire puis agrandir
        small = max(32, int(size * rng.uniform(0.4, 0.75)))
        image = image.resize((small, small), Image.BILINEAR).resize((size, size), Image.BILINEAR)
    # Grain en dernier, comme le bruit d'un capteur de caméra : il couvre aussi les cartes
    array = np.asarray(image, dtype=np.float32) + rng.normal(0, rng.uniform(2, 10), (size, size, 3))
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:  # compression JPEG forte, comme une vidéo ou une photo partagée
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(rng.integers(30, 86)))
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return image


def shade_card(sprite, rng):
    """Carte un peu plus sombre ou plus chaude que le blanc pur, comme un papier sous une lampe."""
    array = np.asarray(sprite, dtype=np.float32).copy()
    gain = rng.uniform(0.8, 1.0) * np.array([1.0, rng.uniform(0.95, 1.0), rng.uniform(0.88, 1.0)])
    array[..., :3] *= gain
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGBA")


def paste_shadow(image, sprite, x, y, rng):
    """Ombre portée sous la carte : un bord sombre et flou, comme sur une vraie table."""
    blur = rng.uniform(1, 5)
    alpha = sprite.getchannel("A").point(lambda v: int(v * rng.uniform(0.3, 0.6)))
    shadow = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(blur)))
    dx, dy = (int(v) for v in rng.integers(-6, 7, 2))
    image.paste(shadow, (x + dx, y + dy), shadow)


# --- Géométrie ---------------------------------------------------------------------------------


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


def random_angle(rng):
    if rng.random() < FULL_ROTATION_PROB:
        return rng.uniform(-180, 180)
    return rng.uniform(-SMALL_ROTATION, SMALL_ROTATION)


def transform_card(sprite, style, rng, height, max_perspective, max_size):
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
    angle = random_angle(rng)
    sprite = sprite.rotate(angle, resample=Image.BICUBIC, expand=True)
    mask = mask.rotate(angle, resample=Image.NEAREST, expand=True)
    bbox = sprite.getchannel("A").getbbox()  # découpe au ras de la carte déformée
    sprite, mask = sprite.crop(bbox), mask.crop(bbox)
    if max(sprite.size) > max_size:  # garde-fou : la carte doit tenir dans l'image
        scale = max_size / max(sprite.size)
        size = (max(1, int(sprite.width * scale)), max(1, int(sprite.height * scale)))
        sprite, mask = sprite.resize(size, Image.LANCZOS), mask.resize(size, Image.NEAREST)
    return sprite, mask


def place_cards(image, deck, rng, n_cards, base_height, max_perspective, max_tries=50):
    """Colle des cartes d'un même jeu au hasard ; elles peuvent se recouvrir.

    deck : (style, {carte: image}), un élément de load_decks().
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
        sprite = shade_card(card_images[card], rng)
        sprite, mask = transform_card(sprite, style, rng, height, max_perspective, size)
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

        if rng.random() < 0.5:
            paste_shadow(image, sprite, x, y, rng)
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


# Probabilité de chaque nombre de cartes visé : 0, 1, 2, 3, 4, 5, 6.
# Les vraies photos de test ont 4 à 5 cartes : on en met souvent 3 à 5.
N_CARDS_PROBS = (0.04, 0.12, 0.16, 0.20, 0.20, 0.16, 0.12)

# Hauteur des cartes, en pixels, pour une image de 384 px. On tire une hauteur par image,
# comme une distance à la caméra : toutes les cartes d'une image ont à peu près la même
# taille. La hauteur maximale baisse avec le nombre de cartes, pour qu'elles tiennent dans
# l'image. Le minimum garde des coins d'environ 10 × 19 px.
MIN_CARD_HEIGHT = 75
MAX_CARD_HEIGHT = {0: 300, 1: 300, 2: 230, 3: 200, 4: 180, 5: 165, 6: 150}

MAX_PERSPECTIVE = 0.25
DECOY_PROB = 0.3


def make_scene(decks, rng, image_size=384, n_cards_probs=N_CARDS_PROBS, max_perspective=MAX_PERSPECTIVE):
    """Retourne (image RGB, liste de cartes, liste des deux coins de chaque carte, nom du jeu).

    Toutes les cartes d'une image viennent du même jeu, comme sur une vraie table.
    Le nombre de cartes posées peut être inférieur au nombre visé : une carte qui ne
    trouve pas de place acceptable est abandonnée.
    """
    image = random_background(rng, image_size)
    n = int(rng.choice(len(n_cards_probs), p=n_cards_probs))
    scale = image_size / 384  # les hauteurs sont données pour 384 px
    base_height = rng.uniform(MIN_CARD_HEIGHT, MAX_CARD_HEIGHT[n]) * scale
    if rng.random() < DECOY_PROB:
        add_decoys(image, rng, base_height)
    deck = decks[int(rng.integers(len(decks)))]
    cards, corners = place_cards(image, deck, rng, n, base_height, max_perspective)
    return augment(image, rng), cards, corners, deck[0].name
