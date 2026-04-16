import calendar
import re

from pandas import Series


def is_valid_date(value: str) -> bool:
    """
    Validate '20YY/MM/DD' with YY in 10..26 and calendar-correct day.

    :param value: The date string to validate.
    :return: True if the date is valid, False otherwise.
    :rtype: bool
    """
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
    """
    Determines whether a given string is a valid time format (HH:MM:SS).

    A valid time format string should match the pattern HH:MM:SS where:
    - HH is a two-digit hour (00-23).
    - MM is a two-digit minute (00-59).
    - SS is a two-digit second (00-60). (60 is allowed for leap seconds)

    The function verifies both the structural format and the numerical ranges
    of hour, minute, and second components.

    :param value: String representation of time to be validated.
    :return: True if the given string is in valid time format, otherwise False.
    :rtype: bool
    """
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
    """
    Validate exposure time with format 'N.DDDD' or 'N.DDDDs' where 0 <= N <= 60.
    :param value: The exposure time string to validate.
    :type value: str
    :return: True if the exposure time is valid, False otherwise.
    :rtype: bool
    """
    if not isinstance(value, str):
        return False

    m = re.fullmatch(r"(\d+)\.(\d{4})s?", value)
    if not m:
        return False

    integer_part = int(m.group(1))
    return 0 <= integer_part <= 60


def is_valid_filename(value: str) -> bool:
    """
    Validate filename consisting of exactly nine digits.
    :param value: The filename string to validate.
    :type value: str
    :return: True if the filename is valid, False otherwise.
    :rtype: bool
    """
    return isinstance(value, str) and re.fullmatch(r"\d{9}", value) is not None


def is_valid_row(row) -> bool:
    """Validate a parsed record row by checking every expected column.
    Designed to work with:
      - pandas.DataFrame.apply(is_valid_row, axis=1)
      - dask.dataframe.DataFrame.apply(is_valid_row, axis=1, meta=(..., bool))

    :param row: A single row of a parsed dataframe.
    :type row: pandas.Series
    :return: True if the row is valid, False otherwise.
    :rtype: bool
    """
    return (
        is_valid_date(row["date"])
        and is_valid_time(row["time"])
        and is_valid_exposure(row["exposure"])
        and is_valid_filename(row["filename"])
    )


def invalid_columns(row) -> list[str]:
    """
    Return the names of invalid columns in a parsed record row.
    :param row: A single row of a parsed dataframe.
    :type row: pandas.Series
    :return: List of column names that are invalid.
    :rtype: list[str]
    """
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


def is_valid_date_series(dataframe) -> Series:
    """
    Validate the datestring column of the parsed dataframe.
    :param dataframe: The parsed dataframe.
    :type dataframe: pandas.DataFrame
    :return: Series of boolean indicating validity of each row.
    :rtype: pandas.Series
    """
    return dataframe["date"].apply(is_valid_date, meta=(..., bool))


def is_valid_time_series(dataframe) -> Series:
    """
    Validate the timestamp column of the parsed dataframe.
    :param dataframe: The parsed dataframe.
    :type dataframe: pandas.DataFrame
    :return: Series of boolean indicating validity of each row.
    :rtype: pandas.Series
    """
    return dataframe["time"].apply(is_valid_time, meta=(..., bool))


def is_valid_exposure_series(dataframe) -> Series:
    """
    Validate the exposure column of the parsed dataframe.
    :param dataframe: The parsed dataframe.
    :type dataframe: pandas.DataFrame
    :return: Series of boolean indicating validity of each row.
    :rtype: pandas.Series
    """
    return dataframe["exposure"].apply(is_valid_exposure, meta=(..., bool))


def is_valid_filename_series(dataframe) -> Series:
    """
    Validate the filename column of the parsed dataframe.
    :param dataframe: The parsed dataframe.
    :type dataframe: pandas.DataFrame
    :return: Series of boolean indicating validity of each row.
    :rtype: pandas.Series
    """
    return dataframe["filename"].apply(is_valid_filename, meta=(..., bool))


def is_valid_record_series(dataframe) -> Series:
    """
    Validate the parsed dataframe.
    :param dataframe: The parsed dataframe.
    :type dataframe: pandas.DataFrame
    :return: Series of boolean indicating validity of each row.
    :rtype: pandas.Series
    """
    return (
        is_valid_date_series(dataframe)
        & is_valid_time_series(dataframe)
        & is_valid_exposure_series(dataframe)
        & is_valid_filename_series(dataframe)
    )
