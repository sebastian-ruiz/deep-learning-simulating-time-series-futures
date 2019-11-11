from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
from helpers.evaluate import *
import numpy as np
import pandas as pd
import hashlib
import json
from classical.AMModel import AMModel
from datetime import datetime as dt
import time


class AndersenMarkovModel:
    def __init__(self):
        print("Andersen Markov Model")

        self.plotting = Plotting()
        preprocess_logreturns = PreprocessData()
        preprocess_logreturns.enable_log_returns = True

        # 1. get data and apply minimax
        sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess_logreturns.get_data()

        # tenors: rate tenors in year fractions (from 0.083 to 5 over 60 steps)
        # rates: corresponding zero rates matrix
        # obs_time: observation dates in year fractions (starting at the first date)
        #           988 steps from -3.835... to 0 on the WTI NYMEX data

        num_c = 6 * 7  # add '* 20' to see if a larger training set helps
        num_o = 6 * 7

        train_set = sets_test[-1].iloc[:num_c]
        test_set = sets_test[-1].iloc[num_c:num_c + num_o + 1]
        num_of_test_curves = len(test_set)

        self.test_set = test_set

        tenors = maturities
        self.tenors = tenors[:, np.newaxis]
        self.rates = np.array(train_set)

        index = pd.Series(train_set.index)
        end_num = toYearFraction(sets_test[-1].index[-1])
        dates_as_decimal = np.array(index.apply(lambda x: toYearFraction(x, end_num)))
        self.dates_as_decimal = dates_as_decimal[:, np.newaxis]
        print("test_set.shape", np.array(test_set).shape)
        smape_results = []
        for i in np.arange(100):

            simulated_rates = self.simulate(num_of_test_curves)

            smape_result = smape(simulated_rates, test_set)
            smape_results.append(smape_result)

            print("simulate rates", i)
            print("simulated, real", np.array(simulated_rates).shape, np.array(test_set).shape)
            print("smape:", smape_result)
            print("=============\n")

            # self.plotting.plot_3d("real", test_set, show_title=False)
            # self.plotting.plot_3d("AMM_simulated_" + str(i), simulated_rates, show_title=False)
            #
            # cov_log_returns = cov_log_returns_over_features(simulated_rates)
            # self.plotting.plot_3d_cov("AMM_simulated_" + str(i) + "_cov", cov_log_returns, show_title=False)

        smape_results = np.array(smape_results)
        # print("smape_results:", smape_results)
        print("smape mean and std:", np.mean(smape_results), np.std(smape_results))


    def simulate(self, num_of_curves=1000):

        self.ammodel = AMModel()
        self.ammodel.set_data(self.tenors, self.rates, self.dates_as_decimal, None)
        self.ammodel.calibrate_moments()
        self.ammodel.make_parameters()

        self.ammodel.make_data(m=num_of_curves)

        simulated_rates = self.ammodel.rates

        # print(simulated_rates, simulated_rates.shape)

        # self.plotting.plot_3d("AMModel_test", simulated_rates)

        return pd.DataFrame(simulated_rates, index=self.test_set.index.values, columns=self.test_set.columns.values)

def toYearFraction(date, start_num=None):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    if start_num != None:
        return date.year + fraction - start_num
    else:
        return date.year + fraction


if __name__ == '__main__':
    andersenMarkovModel = AndersenMarkovModel()

