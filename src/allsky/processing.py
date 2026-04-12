import os

from dask.array.image import imread


def process_allsky_image_folder(directory: str, classify_fields_block):
    """
    Process all images in a directory and classify fields using a provided block function.

    :param directory: Directory containing images to process.
    :param classify_fields_block: Block function for classifying fields in images.
    """
    filename_pattern = f"{directory}*.JPG"
    images = imread(filename_pattern)
    # TODO: rework the transpose for aliased parse to use the same process as the antialiased one
    all_batch_reds = images.transpose(0, 3, 1, 2)[:, 0].rechunk((32, 480, 640))
    df = all_batch_reds.map_blocks(
        classify_fields_block, dtype=object, drop_axis=(1, 2)
    ).compute()
    df.to_parquet(f"../../data/{os.path.basename(directory[:-1])}.parquet")
    print("Done: ", directory)
