from skyfield.api import load
from skyfield.api import N, W, wgs84
from skyfield.api import Time as SkyfieldTime
from skyfield.api import Timescale
from skyfield.api import Angle, Distance
from skyfield.vectorlib import VectorSum
from skyfield import almanac
from pytz import timezone
import pandas as pd
import dask.dataframe as dd
import numpy as np

ASTRONOMICAL_TWILIGHT = -18
NEWMOON_LIGHT_FRACTION = 0.05

ts = load.timescale(builtin=True)
planets = load('de405.bsp')

moon: VectorSum = planets['moon']
sun = planets['sun']
earth = planets['earth']

allegheny: VectorSum = earth + wgs84.latlon(40.48250427897144 * N, 80.02063960912749 * W, 372.86)
allegheny_tz = timezone('US/Eastern')

def get_altaz(target, ts: SkyfieldTime) -> tuple[Angle, Angle]:
    alt, az, _ = allegheny.at(ts).observe(target).apparent().altaz()
    return alt.degrees, az.degrees

def get_moon_phase(ts: SkyfieldTime) -> np.ndarray[np.float32]:
    return allegheny.at(ts).observe(moon).apparent().fraction_illuminated(sun)

def get_times_from_dataframe(dataframe: pd.DataFrame | dd.DataFrame) -> SkyfieldTime:
    timestamps = dataframe["timestamp"].apply(allegheny_tz.localize)
    return ts.from_datetimes(timestamps)

