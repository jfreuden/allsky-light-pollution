import warnings

warnings.warn(
    "allsky.ocr is experimental and may change without notice.",
    DeprecationWarning,
    stacklevel=2,
)

import os
from PIL import Image
from PIL.Image import Resampling
import pyocr.builders

ocr_tools = pyocr.get_available_tools()
cmd_tesseract = ocr_tools[0]
lib_tesseract = ocr_tools[1]

import numpy as np

def get_statblock_cutout(image: Image.Image) -> Image.Image:
    full_height = 480
    block_width = 80
    top_height = 54
    bottom_height = 20

    top_block = image.crop((0, 0, block_width, top_height))
    top_block = top_block.convert('L')
    bottom_block = image.crop((0, full_height - bottom_height, block_width, full_height))
    bottom_block = bottom_block.convert('L')

    stat_block = Image.new('RGB', (block_width, top_height + bottom_height))
    stat_block.paste(top_block, (0, 0))
    stat_block.paste(bottom_block, (0, top_height))
    stat_block = stat_block.resize((3 * block_width, 2 * (top_height + bottom_height)), Resampling.LANCZOS)

    return stat_block

def process_one_image(method, item: np.ndarray):
    image = Image.fromarray(item)
    stat_block = get_statblock_cutout(image)
    txt = method.image_to_string(stat_block, lang='eng', builder=pyocr.builders.DigitBuilder())
    return txt

def process_one_block(method, block:np.ndarray):
    return np.asarray([process_one_image(method, item) for item in block], dtype=object)

def process_with_cmd(block: np.ndarray):
    return process_one_block(cmd_tesseract, block)

def process_with_lib(block: np.ndarray):
    return process_one_block(lib_tesseract, block)