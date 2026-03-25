from allsky.classifiers import classify_fields_block

import os
from dask.array.image import imread

def process_allsky_image_folder(directory: str,):
    filename_pattern = f'{directory}*.JPG'
    images = imread(filename_pattern)
    # TODO: see if I can rework the downstream code to not need this transpose
    all_batch_reds = images.transpose(0, 3, 1, 2)[:,0].rechunk((32,480,640))
    df = all_batch_reds.map_blocks(classify_fields_block, dtype=object, drop_axis=(1,2)).compute()
    df.to_parquet(f'../../data/{os.path.basename(directory[:-1])}.parquet')
    print("Done: ", directory)