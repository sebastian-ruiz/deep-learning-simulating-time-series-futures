import numpy as np
import pandas as pd
import datetime

def convert_dates_to_days(dates, start_date=None, name='Day'):
    """Converts a series of dates to a series of float values that
    represent days since start_date.
    """

    if start_date:
        ts0 = pd.Timestamp(start_date).timestamp()
    else:
        ts0 = 0

    if isinstance(dates, datetime.date):
        return round((pd.Timestamp(dates).timestamp() - ts0) / (24 * 3600))
    if type(dates) is np.ndarray:
        return [round((pd.Timestamp(date).timestamp() - ts0) / (24 * 3600)) for date in dates]
    else:
        return round(((dates.apply(pd.Timestamp.timestamp) - ts0) / (24 * 3600)).rename(name))
