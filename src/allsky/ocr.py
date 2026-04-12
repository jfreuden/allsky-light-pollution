import warnings

warnings.warn(
    "allsky.ocr is experimental and may change without notice.",
    DeprecationWarning,
    stacklevel=2,
)

import pyocr.builders
from PIL import Image
from PIL.Image import Resampling

ocr_tools = pyocr.get_available_tools()
cmd_tesseract = ocr_tools[0]
lib_tesseract = ocr_tools[1]

import numpy as np


def get_statblock_cutout(image: Image.Image) -> Image.Image:
    """
    Extracts a standardized stat block cutout from a full image.

    :param image: The full image as a PIL Image.
    :return: A cropped and resized stat block image.
    """
    full_height = 480
    block_width = 80
    top_height = 54
    bottom_height = 20

    top_block = image.crop((0, 0, block_width, top_height))
    top_block = top_block.convert("L")
    bottom_block = image.crop(
        (0, full_height - bottom_height, block_width, full_height)
    )
    bottom_block = bottom_block.convert("L")

    stat_block = Image.new("RGB", (block_width, top_height + bottom_height))
    stat_block.paste(top_block, (0, 0))
    stat_block.paste(bottom_block, (0, top_height))
    stat_block = stat_block.resize(
        (3 * block_width, 2 * (top_height + bottom_height)), Resampling.LANCZOS
    )

    return stat_block


def process_one_image(method, item: np.ndarray):
    """
    Process a single image using the specified OCR method.

    :param method: The OCR method to use (e.g., cmd_tesseract, lib_tesseract).
    :param item: The image as a NumPy array.
    :return: The extracted text from the stat block.
    """
    image = Image.fromarray(item)
    stat_block = get_statblock_cutout(image)
    txt = method.image_to_string(
        stat_block, lang="eng", builder=pyocr.builders.DigitBuilder()
    )
    return txt


def process_one_block(method, block: np.ndarray):
    """
    Process a block of images using the specified OCR method.

    :param method: The OCR method to use (e.g., cmd_tesseract, lib_tesseract).
    :param block: A list of images as NumPy arrays.
    :return: A list of extracted text from stat blocks for each image in the block.
    """
    return np.asarray([process_one_image(method, item) for item in block], dtype=object)


def process_with_cmd(block: np.ndarray):
    """
    Process a block of images using the command-line Tesseract OCR.

    :param block: A list of images as NumPy arrays.
    :return: A list of extracted text from stat blocks for each image in the block.
    """
    return process_one_block(cmd_tesseract, block)


def process_with_lib(block: np.ndarray):
    """
    Process a block of images using the Tesseract libtesseract OCR library.
    This yields generally better performance.

    :param block: A list of images as NumPy arrays.
    :return: A list of extracted text from stat blocks for each image in the block.
    """
    return process_one_block(lib_tesseract, block)
