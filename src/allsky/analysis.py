import dask.dataframe as dd
import numpy as np
import pandas as pd


def quick_exposure_linregress(
    dataframe: pd.DataFrame | dd.DataFrame, y_variable="exposure"
) -> tuple[float, float]:
    """
    Returns a tuple of (slope, intercept) of a linear regression of the given variable, defaulting to exposure.

    Parameters:
    - dataframe: The input dataframe, either pandas or dask.
    - y_variable: The variable to perform the linear regression on. Defaults to 'exposure'.

    Returns:
    - tuple[float, float]: A tuple containing the slope and intercept of the linear regression.
    """
    x = dataframe["timestamp"].map(pd.Timestamp.toordinal).to_numpy()
    y = dataframe[y_variable].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


y, x = np.ogrid[:480, :640]
mask_2d = (x - 320) ** 2 + (y - 240) ** 2 > 200**2


def mask_image(image) -> np.ma.MaskedArray:
    """
    Masks an allsky image by applying a circular mask and cropping to a specific region.

    Parameters:
    - image: The input image as a 2D or 3D numpy array. (This expects a 640x480 image)

    Returns:
    - np.ma.MaskedArray: The masked and cropped image with a circular mask centered at
        (320, 240) with a radius of 200 pixels
    """
    if image.ndim == 2:
        out = np.ma.array(image, mask=mask_2d)
    elif image.ndim == 3:
        mask_3d = np.broadcast_to(mask_2d[:, :, None], image.shape)
        out = np.ma.array(image, mask=mask_3d)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return out[40:440, 120:520]
