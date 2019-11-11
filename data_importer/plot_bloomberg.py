import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import datetime
from scipy.interpolate import interp1d
from helpers import *


def main():
    pd.options.mode.chained_assignment = None

    start_date = '2018-01-01'
    end_date = '2018-12-31'

    df = pd.read_csv('testdata.txt',
                     sep='|',
                     header=0,
                     names=['gen_con_names', 'hist_dates', 'prices', 'sp_con_names', 'exp_dates'],
                     parse_dates=['hist_dates', 'exp_dates'],
                     dayfirst=True,
                     decimal=',')

    # business days starting at beginning of year, right now only look at first few days while testing
    num_of_days = 50
    dates = pd.Series(pd.date_range(start=start_date, periods=num_of_days, freq='B'))
    dates_as_days = convert_dates_to_days(dates, start_date=start_date)

    # split year into 12 periods where we take take readings
    intervals = pd.Series(pd.date_range(start_date, end_date, freq='M'))
    intervals_as_days = convert_dates_to_days(intervals, start_date=start_date)

    surface = []

    for (date, date_as_day) in zip(dates, dates_as_days):
        temp = df.loc[(df['hist_dates'] == date)]
        if len(temp) > 0:
            # add start and end points
            temp.loc[-1] = ['', '', temp.iloc[-1]['prices'], '', datetime.datetime.strptime(end_date, '%Y-%m-%d')]
            temp.loc[-2] = ['', '', temp.iloc[0]['prices'], '', datetime.datetime.strptime(start_date, '%Y-%m-%d')]
            temp = temp.sort_values(by='exp_dates')

            x = convert_dates_to_days(temp.exp_dates, start_date=start_date)
            y = temp.prices
            f2 = interp1d(x, y, kind='linear', axis=0, fill_value="extrapolate")

            all_dates = pd.Series(pd.date_range(start_date, end_date))
            all_dates_as_days = convert_dates_to_days(all_dates, start_date=start_date)

            surface.append(f2(intervals_as_days + date_as_day - 1)) # subtract 1 since we do not shift the first day

    surface = np.array(surface)

    print(surface)
    print(surface.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xx, yy = np.meshgrid(intervals_as_days, np.arange(len(surface)))
    ax.plot_surface(xx, yy, surface, cmap=cm.coolwarm)
    ax.set_xlabel('maturity')
    ax.set_ylabel('t')
    ax.set_zlabel('price')
    fig.savefig('plot.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
