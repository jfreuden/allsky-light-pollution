import dask.dataframe as dd
import numpy as np
import pandas as pd
from pytz import timezone
from skyfield.api import Angle, Distance, N
from skyfield.api import Time as SkyfieldTime
from skyfield.api import Timescale, W, load, wgs84
from skyfield.vectorlib import VectorSum

ASTRONOMICAL_TWILIGHT = -18
NEWMOON_LIGHT_FRACTION = 0.05

ts = load.timescale(builtin=True)
planets = load("de405.bsp")

moon: VectorSum = planets["moon"]
sun = planets["sun"]
earth = planets["earth"]

allegheny: VectorSum = earth + wgs84.latlon(
    40.48250427897144 * N, 80.02063960912749 * W, 372.86
)
allegheny_tz = timezone("US/Eastern")


def get_altaz(target, ts: SkyfieldTime) -> tuple[Angle, Angle]:
    """
    Calculates the altitude and azimuth of a target relative to Allegheny Observatory at a given time.

    Parameters:
    - target: The celestial body to observe.
    - ts: The time at which to observe the target.

    Returns:
    - tuple[Angle, Angle]: A tuple containing the altitude and azimuth angles in degrees.
    """
    alt, az, _ = allegheny.at(ts).observe(target).apparent().altaz()
    return alt.degrees, az.degrees


def get_moon_phase(ts: SkyfieldTime) -> np.ndarray[np.float32]:
    """
    Calculates the moon phase for the Allegheny Observatory at a given time.

    Parameters:
    - ts: The time at which to calculate the moon phase.

    Returns:
    - np.ndarray[np.float32]: An array of moon phase fractions.
    """
    return allegheny.at(ts).observe(moon).apparent().fraction_illuminated(sun)


def get_times_from_dataframe(dataframe: pd.DataFrame | dd.DataFrame) -> SkyfieldTime:
    """
    Converts a dataframe of timestamps to Skyfield Time objects for Allegheny Observatory.

    Parameters:
    - dataframe: The input dataframe containing timestamps.

    Returns:
    - SkyfieldTime: A Skyfield Time object representing the timestamps loaded into the timescale.
    """
    timestamps = dataframe["timestamp"].apply(allegheny_tz.localize)
    return ts.from_datetimes(timestamps)
