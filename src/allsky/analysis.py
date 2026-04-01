import numpy as np
import pandas as pd
import dask.dataframe as dd
def quick_exposure_linregress(dataframe: pd.DataFrame | dd.DataFrame, y_variable="exposure") -> tuple[float, float]:
    x = dataframe["timestamp"].map(pd.Timestamp.toordinal).to_numpy()
    y = dataframe[y_variable].to_numpy()
    return np.polyfit(x, y, 1)