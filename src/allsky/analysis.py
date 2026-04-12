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
