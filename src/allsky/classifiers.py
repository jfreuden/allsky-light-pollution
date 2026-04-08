import numpy as np
import pandas as pd
from dask.array.image import imread

# These are specific to my atlases, but shared between both aliased and antialiased ones
char_width = 6
char_height = 8
symb_width = 3
symb_height = 8
atlas_chars = "0123456789s"
atlas_symbs = ":/."

# all of these are specific to my atlas, particularly the aliased one. They are overridden in `classifiers_antialiased.py`
atlas_image = imread('../../images/font_atlas.bmp') # 66 x 16 pixels

# Now split into two atlas char arrays, one for the characters at 6x8 pixels and one for symbols at 3x8
# Note: the symbol atlas is not used in the current implementation, as we know where all the symbols should be and can fail the parse in the cases when we are wrong.
char_atlas = atlas_image[0][:8, :]
symb_atlas = atlas_image[0][8:, :]

atlas_charcount = char_atlas.shape[1] // char_width

# Reshape into (8, chars, char_width)
char_atlas_chars = char_atlas.reshape(8, atlas_charcount, char_width).transpose(1, 0, 2).astype(np.float32).compute()
char_glyphs = dict(zip(atlas_chars, char_atlas_chars))

digit_templates = np.stack([char_glyphs[ch] for ch in atlas_chars]).astype(np.float32)
digit_templates = digit_templates - digit_templates.mean(axis=(1, 2), keepdims=True)
digit_template_norms = np.sqrt(np.sum(digit_templates * digit_templates, axis=(1, 2))) + 1e-6
digit_templates = digit_templates.reshape(digit_templates.shape[0], -1)

decimal_template = symb_atlas[:, 6:9].astype(np.float32)
decimal_template = decimal_template - decimal_template.mean(keepdims=True)
decimal_template_norm = np.sqrt(np.sum(decimal_template * decimal_template)) + 1e-6

def score_patch_against_template(patch, template, eps=1e-6):
    """
    Returns a normalized correlation score between a patch and a glyph template.
    Higher is better.
    """
    patch = np.asarray(patch, dtype=np.float32)
    template = np.asarray(template, dtype=np.float32)

    patch = patch - patch.mean()
    template = template - template.mean()

    denom = np.sqrt(np.sum(patch ** 2) * np.sum(template ** 2)) + eps
    return float(np.sum(patch * template) / denom)


def classify_at_cursor(image, x, y, atlas, atlas_chars, width, height, eps=1e-6) -> tuple[str, float, dict[str, float]]:
    """
    Classify the glyph centered/anchored at a known cursor position.

    Parameters
    ----------
    image : ndarray
        2D grayscale image.
    x, y : int
        Top-left position of the glyph box in the image.
    atlas : dict[str, ndarray]
        Mapping from character label to glyph bitmap.
    atlas_chars : iterable[str]
        Characters to test.
    width, height : int
        Glyph size.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    best_char : str
    best_score : float
    scores : dict[str, float]
    """
    patch = image[y:y + height, x:x + width]
    if patch.shape != (height, width) and patch.shape != (height, width, 3):
        raise ValueError(f"Patch shape {patch.shape} does not match {(height, width)}")

    scores = {}
    for ch in atlas_chars:
        scores[ch] = score_patch_against_template(patch, atlas[ch], eps=eps)

    best_char = max(scores, key=scores.get)
    best_score = scores[best_char]
    return best_char, best_score, scores

def extract_patches_2d(image, positions, width, height):
    patches = []
    for x, y in positions:
        patch = np.asarray(image[y:y + height, x:x + width], dtype=np.float32)
        if patch.shape != (height, width) and patch.shape != (height, width, 3):
            raise ValueError(f"Bad patch shape {patch.shape}, expected {(height, width)}")
        patch = patch - patch.mean()
        patches.append(patch.reshape(-1))
    return np.stack(patches, axis=0)

def classify_patches_2d(patches_2d, templates_2d, template_norms):
    patch_norms = np.linalg.norm(patches_2d, axis=1) + 1e-6
    raw = patches_2d @ templates_2d.T
    scores = raw / (patch_norms[:, None] * template_norms[None, :])

    best_idx = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(scores.shape[0]), best_idx]
    return best_idx, best_scores, scores


# Represents the positions "XXXX/XX/XX" in the top-left, first row
DATE_POS = [
    (5 + 0 * char_width, 6),
    (5 + 1 * char_width, 6),
    (5 + 2 * char_width, 6),
    (5 + 3 * char_width, 6),
    (5 + 4 * char_width + symb_width, 6),
    (5 + 5 * char_width + symb_width, 6),
    (5 + 6 * char_width + 2 * symb_width, 6),
    (5 + 7 * char_width + 2 * symb_width, 6),
]

# Represents the positions "XX:XX:XX" in the top-left, second row
TIME_POS = [
    (5 + 0 * char_width, 19),
    (5 + 1 * char_width, 19),
    (20 + 0 * char_width, 19),
    (20 + 1 * char_width, 19),
    (35 + 0 * char_width, 19),
    (35 + 1 * char_width, 19),
]

# Represents the positions "XXXXXXXXX" in the bottom-left.
FILENAME_POS = [
    (5 + 0 * char_width, 466),
    (5 + 1 * char_width, 466),
    (5 + 2 * char_width, 466),
    (5 + 3 * char_width, 466),
    (5 + 4 * char_width, 466),
    (5 + 5 * char_width, 466),
    (5 + 6 * char_width, 466),
    (5 + 7 * char_width, 466),
    (5 + 8 * char_width, 466),
]

IMAGE_DIMENSIONS = (480, 640)

def classify_patches_deterministic(image, positions, tolerance: float):
    patches_2d = extract_patches_2d(image, positions, char_width, char_height)
    best_idx, best_scores, _ = classify_patches_2d(patches_2d, digit_templates, digit_template_norms)

    chars = []
    for idx, score in zip(best_idx, best_scores):
        ch = atlas_chars[idx]
        if score < tolerance:
            ch = '?'
        chars.append(ch)
    return chars


def classify_date_string(image, tolerance=0.85):
    image = np.asarray(image, dtype=np.float32).reshape(IMAGE_DIMENSIONS)
    chars = classify_patches_deterministic(image, DATE_POS, tolerance)

    chars = chars[:4] + ['/'] + chars[4:6] + ['/'] + chars[6:]
    return ''.join(chars)

def classify_time_string(image, tolerance=0.85):
    image = np.asarray(image, dtype=np.float32).reshape(IMAGE_DIMENSIONS)
    chars = classify_patches_deterministic(image, TIME_POS, tolerance)
    chars = chars[:2] + [':'] + chars[2:4] + [':'] + chars[4:]
    return ''.join(chars)

EXPOSURE_DECIMAL_POS = [
    (5 + 1 * char_width, 32),
    (5 + 2 * char_width, 32),
]

def classify_exposure_string(image, tolerance=0.85):
    image = np.asarray(image, dtype=np.float32).reshape(IMAGE_DIMENSIONS)

    # This cannot be done deterministically. I must find the number of digits before the decimal point first.
    for preceding_digits, candidate_decimal_pos in enumerate(EXPOSURE_DECIMAL_POS,1):
        _, score, _ = classify_at_cursor(
            image,
            *candidate_decimal_pos,
            atlas={'.': decimal_template},
            atlas_chars=['.'],
            width=symb_width,
            height=symb_height,
        )
        if score > tolerance:
            # We found the decimal with `preceding_digits` before, and four after.
            predecimal_positions = [
                (5 + x * char_width, 32) for x in range(0, preceding_digits)
            ]
            postdecimal_offset = 5 + preceding_digits * char_width + symb_width
            postdecimal_positions = [
                (postdecimal_offset + x * char_width, 32) for x in range(0, 4)
            ]
            predecimal_chars = classify_patches_deterministic(image, predecimal_positions, tolerance)
            postdecimal_chars = classify_patches_deterministic(image, postdecimal_positions, tolerance)

            return ''.join(predecimal_chars) + '.' + ''.join(postdecimal_chars)
    # If we fell out of the loop because we didn't find the decimal, we can't classify the exposure time.
    return None

def classify_filename_string(image, tolerance=0.85):
    image = np.asarray(image, dtype=np.float32).reshape(IMAGE_DIMENSIONS)
    chars = classify_patches_deterministic(image, FILENAME_POS, tolerance)
    return ''.join(chars)

def classify_fields_block(block):
    results = []
    for img in block:
        results.append({
            'date': classify_date_string(img),
            'time': classify_time_string(img),
            'exposure': classify_exposure_string(img),
            'filename': classify_filename_string(img),
            # TODO: fields related to the image content
        })
    return pd.DataFrame.from_records(results)
