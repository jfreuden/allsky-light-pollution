import calendar
import re
from narwhals import Series


def is_valid_date(value: str) -> bool:
    """Validate '20YY/MM/DD' with YY in 10..26 and calendar-correct day."""
    if not isinstance(value, str):
        return False

    m = re.fullmatch(r"20(\d{2})/(\d{2})/(\d{2})", value)
    if not m:
        return False

    yy = int(m.group(1))
    mm = int(m.group(2))
    dd = int(m.group(3))

    if not (10 <= yy <= 26 and 1 <= mm <= 12):
        return False

    year = 2000 + yy
    max_day = calendar.monthrange(year, mm)[1]
    return 1 <= dd <= max_day


def is_valid_time(value: str) -> bool:
    """Validate 'HH:MM:SS' with HH 00..23, MM 00..59, SS 00..60."""
    if not isinstance(value, str):
        return False

    m = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})", value)
    if not m:
        return False

    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = int(m.group(3))

    return 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 60


def is_valid_exposure(value: str) -> bool:
    """Validate exposure time with format 'N.DDDD' or 'N.DDDDs' where 0 <= N <= 30."""
    if not isinstance(value, str):
        return False

    m = re.fullmatch(r"(\d+)\.(\d{4})s?", value)
    if not m:
        return False

    integer_part = int(m.group(1))
    return 0 <= integer_part <= 60


def is_valid_filename(value: str) -> bool:
    """Validate filename consisting of exactly nine digits."""
    return isinstance(value, str) and re.fullmatch(r"\d{9}", value) is not None

def is_valid_row(row) -> bool:
    """Validate a parsed record row by checking every expected column.
    Designed to work with:
      - pandas.DataFrame.apply(is_valid_row, axis=1)
      - dask.dataframe.DataFrame.apply(is_valid_row, axis=1, meta=(..., bool))
    """
    return (
        is_valid_date(row["date"])
        and is_valid_time(row["time"])
        and is_valid_exposure(row["exposure"])
        and is_valid_filename(row["filename"])
    )

def invalid_columns(row) -> list[str]:
    """Return the names of invalid columns in a parsed record row."""
    invalid = []

    if not is_valid_date(row["date"]):
        invalid.append("date")
    if not is_valid_time(row["time"]):
        invalid.append("time")
    if not is_valid_exposure(row["exposure"]):
        invalid.append("exposure")
    if not is_valid_filename(row["filename"]):
        invalid.append("filename")

    return invalid

def is_valid_date_series(dataframe) -> Series[bool]:
    """Validate the datestring column of the parsed dataframe."""
    return dataframe["date"].apply(is_valid_date, meta=(..., bool))

def is_valid_time_series(dataframe) -> Series[bool]:
    """Validate the timestamp column of the parsed dataframe."""
    return dataframe["time"].apply(is_valid_time, meta=(..., bool))

def is_valid_exposure_series(dataframe) -> Series[bool]:
    """Validate the exposure column of the parsed dataframe."""
    return dataframe["exposure"].apply(is_valid_exposure, meta=(..., bool))

def is_valid_filename_series(dataframe) -> Series[bool]:
    """Validate the filename column of the parsed dataframe."""
    return dataframe["filename"].apply(is_valid_filename, meta=(..., bool))

def is_valid_record_series(dataframe) -> Series[bool]:
    """Validate the parsed dataframe."""
    return (
        is_valid_date_series(dataframe) &
        is_valid_time_series(dataframe) &
        is_valid_exposure_series(dataframe) &
        is_valid_filename_series(dataframe)
    )
