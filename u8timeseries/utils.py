from dateutils import relativedelta
import pandas as pd


def add_time_delta_to_datetime(np_dt, i, stepduration_str):
    """
    Adds [i] time-steps (whose length is given by [stepduration_str]) to the numpy datetime [np_dt]
    """
    datet = pd.Timestamp(np_dt)

    switch = {
        'second': lambda dt: dt + relativedelta(seconds=+i),
        'minute': lambda dt: dt + relativedelta(minutes=+i),
        'hour': lambda dt: dt + relativedelta(hours=+i),
        'day': lambda dt: dt + relativedelta(days=+i),
        'week': lambda dt: dt + relativedelta(weeks=+i),
        'month': lambda dt: dt + relativedelta(months=+i),
        'quarter': lambda dt: dt + relativedelta(months=+3*i),
        'year': lambda dt: dt + relativedelta(years=+i)
    }
    return switch[stepduration_str](datet)


def fill_dates_between(first_dt, last_dt, stepduration_str):
    """
    Returns a list of all dates between [first_dt] and [last_dt] (Numpy datetimes or datetimes) included,
    spaced by [stepduration_str]
    """
    current_date = first_dt
    dates = [current_date]

    while current_date < last_dt:
        current_date = add_time_delta_to_datetime(current_date, 1, stepduration_str)
        if current_date <= last_dt:
            dates.append(current_date)

    return dates
