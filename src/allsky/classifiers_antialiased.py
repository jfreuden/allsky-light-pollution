import allsky.classifiers as classifiers
from dask.array.image import imread
import numpy as np

char_width = classifiers.char_width
char_height = classifiers.char_height
symb_width = classifiers.symb_width
symb_height = classifiers.symb_height
atlas_chars = classifiers.atlas_chars
atlas_symbs = classifiers.atlas_symbs

atlas_image = None
char_atlas = None
symb_atlas = None
atlas_charcount = None
char_atlas_chars = None
char_glyphs = None
digit_templates = None
digit_template_norms = None
decimal_template = None
decimal_template_norm = None
IMAGE_DIMENSIONS = (480, 640, 3)

def initialize_antialiased_classifiers():
    global atlas_image, char_atlas, symb_atlas, atlas_charcount
    global char_atlas_chars, char_glyphs, digit_templates, digit_template_norms
    global decimal_template, decimal_template_norm

    if atlas_image is not None:
        return

    atlas_image = imread('/home/rainybyte/AllSkyImages/antialias-atlas.bmp')
    char_atlas = atlas_image[0][:8, :]
    symb_atlas = atlas_image[0][8:, :]
    atlas_charcount = char_atlas.shape[1] // char_width

    char_atlas_chars = char_atlas.reshape(8, atlas_charcount, char_width, 3).transpose(1, 0, 2, 3).astype(np.float32).compute()
    char_glyphs = dict(zip(atlas_chars, char_atlas_chars))

    digit_templates = np.stack([char_glyphs[ch] for ch in atlas_chars]).astype(np.float32)
    digit_templates = digit_templates - digit_templates.mean(axis=(1, 2, 3), keepdims=True)
    digit_template_norms = np.sqrt(np.sum(digit_templates * digit_templates, axis=(1, 2, 3))) + 1e-6
    digit_templates = digit_templates.reshape(digit_templates.shape[0], -1)

    decimal_template = symb_atlas[:, 6:9].astype(np.float32)
    decimal_template = decimal_template - decimal_template.mean(keepdims=True)
    decimal_template_norm = np.sqrt(np.sum(decimal_template * decimal_template)) + 1e-6

    classifiers.atlas_image = atlas_image
    classifiers.char_atlas = char_atlas
    classifiers.symb_atlas = symb_atlas
    classifiers.atlas_charcount = atlas_charcount
    classifiers.char_atlas_chars = char_atlas_chars
    classifiers.char_glyphs = char_glyphs
    classifiers.digit_templates = digit_templates
    classifiers.digit_template_norms = digit_template_norms
    classifiers.decimal_template = decimal_template
    classifiers.decimal_template_norm = decimal_template_norm
    classifiers.IMAGE_DIMENSIONS = IMAGE_DIMENSIONS

initialize_antialiased_classifiers()

score_patch_against_template = classifiers.score_patch_against_template
classify_at_cursor = classifiers.classify_at_cursor
extract_patches_2d = classifiers.extract_patches_2d
classify_patches_2d = classifiers.classify_patches_2d
classify_patches_deterministic = classifiers.classify_patches_deterministic
classify_date_string = classifiers.classify_date_string
classify_time_string = classifiers.classify_time_string
classify_exposure_string = classifiers.classify_exposure_string
classify_filename_string = classifiers.classify_filename_string
classify_fields_block = classifiers.classify_fields_block

DATE_POS = classifiers.DATE_POS
TIME_POS = classifiers.TIME_POS
FILENAME_POS = classifiers.FILENAME_POS
EXPOSURE_DECIMAL_POS = classifiers.EXPOSURE_DECIMAL_POS