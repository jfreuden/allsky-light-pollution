import numpy as np
import pandas as pd
import dask.dataframe as dd
def quick_exposure_linregress(dataframe: pd.DataFrame | dd.DataFrame) -> tuple[float, float]:
    x = dataframe["timestamp"].map(pd.Timestamp.toordinal).to_numpy()
    y = dataframe["exposure"].to_numpy()
    return np.polyfit(x, y, 1)