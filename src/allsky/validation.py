import calendar
import re

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