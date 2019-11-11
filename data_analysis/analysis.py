from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from helpers.evaluate import *


class Analysis():

    def __init__(self):
        self.preprocess_data = PreprocessData()
        self.plotting = Plotting()
        self.config = Config()

        # self.preprocess_data.enable_min_max_scaler = True
        self.preprocess_data.enable_log_returns = True
        self.sets_training, self.sets_test, self.sets_training_scaled, self.sets_test_scaled, \
        self.training_dataset_names, self.test_dataset_names, self.maturities = self.preprocess_data.get_data()

        wti_nymex = self.sets_test[0]
        time = wti_nymex.axes[0].tolist()

        self.wti_nymex_short_end = wti_nymex.iloc[:, 0]
        self.data_scaled = self.sets_test_scaled[0][0]

    def normalisation_over_tenors(self):
        preprocess = PreprocessData(PreprocessType.NORMALISATION_OVER_TENORS)
        sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess.get_data()

        print("sets_test[0].shape", sets_test[0].shape, sets_test_scaled[0].shape)

        self.plotting.plot_some_curves("normalisation_over_tenors", sets_test[0], sets_test_scaled[0],
                                  [25, 50, 75, 815], maturities, plot_separate=True)  # old: [25, 50, 75, 100, 600, 720, 740, 815]

    def standardisation_over_tenors(self):
        preprocess = PreprocessData(PreprocessType.STANDARDISATION_OVER_TENORS)
        sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess.get_data()

        self.plotting.plot_some_curves("standardisation_over_tenors", sets_test[0], sets_test_scaled[0],
                                       [25, 50, 75, 815], maturities, plot_separate=True)  # old: [25, 50, 75, 100, 600, 720, 740, 815]

    def logreturns_over_tenors(self):
        preprocess = PreprocessData(PreprocessType.LOG_RETURNS_OVER_TENORS)
        sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess.get_data()

        self.plotting.plot_some_curves("logreturns_over_curves", sets_test[0], sets_test_scaled[0],
                                       [25, 50, 75, 815], maturities, plot_separate=True)  # old: [25, 50, 75, 100, 600, 720, 740, 815]

        self.plotting.plot_3d("logreturns_over_curves_3d", sets_test_scaled[0], )

    def normalisation_over_curves(self):
        preprocess = PreprocessData()
        preprocess.enable_normalisation_scaler = True
        preprocess.enable_ignore_price = True
        preprocess.feature_range = [0, 1]
        sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess.get_data()

        self.plotting.plot_some_curves("normalisation_over_curves", sets_test[0], sets_test_scaled[0],
                                       [25, 50, 75, 815], maturities, plot_separate=True)  # old: [25, 50, 75, 100, 600, 720, 740, 815]

    def standardisation_over_curves(self):
        print("todo standardisation_over_curves")

    def logreturns_over_curves(self):
        print("todo logreturns_over_curves")

    def all_log_returns(self):
        preprocess_data = PreprocessData()
        plotting = Plotting()

        preprocess_data.enable_log_returns = True
        sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess_data.get_data()
        for i, set_training_scaled in enumerate(sets_training_scaled):
            print("set_training_scaled.shape", set_training_scaled.shape, i)
            plotting.plot_2d(set_training_scaled, "/time_series/" + training_dataset_names[i], timeseries=True, save=False, title=True)

    def all_normalised_data(self):
        preprocess_data = PreprocessData()

        preprocess_data.enable_normalisation_scaler = True
        sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess_data.get_data()

        for i, set_training_scaled in enumerate(sets_training_scaled):
            self.plotting.plot_2d(set_training_scaled, "/time_series/" + training_dataset_names[i], timeseries=True,
                             save=True, title=True)

        for i, set_test_scaled in enumerate(sets_test_scaled):
            self.plotting.plot_2d(set_test_scaled, "/time_series/" + test_dataset_names[i], timeseries=True, save=True,
                             title=True)

    def all_data(self, show_title=False):
        preprocess_data = PreprocessData(extend_data=False)
        sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess_data.get_data()

        print("maturities", maturities)

        for i, set_training in enumerate(sets_training):
            print(self.training_dataset_names[i])
            print(set_training.index[0], set_training.index[-1], round(np.min(set_training.min()), 2), round(np.max(set_training.max()), 2))
            # self.plotting.plot_2d(set_training, "/time_series/" + training_dataset_names[i], timeseries=True,
            #                  save=True, title=show_title)

            # self.plotting.plot_3d("/time_series/" + training_dataset_names[i] + "_3d", set_training, show_title=show_title)

            cov_log_returns = cov_log_returns_over_tenors(set_training)
            # self.plotting.plot_3d_cov("/time_series/" + training_dataset_names[i] + "_cov", cov_log_returns, maturities=maturities, show_title=show_title)

            print("\n")

        for i, set_test in enumerate(sets_test):
            print(self.test_dataset_names[i])
            print(set_test.index[0], set_test.index[-1], round(np.min(set_test.min()), 2), round(np.max(set_test.max()), 2))
            self.plotting.plot_2d(set_test, "/time_series/" + test_dataset_names[i], timeseries=True, save=True,
                             title=show_title)
            self.plotting.plot_3d("/time_series/" + test_dataset_names[i] + "_3d", set_test, show_title=show_title)

            cov_log_returns = cov_log_returns_over_tenors(set_test)
            # self.plotting.plot_3d_cov("/time_series/" + test_dataset_names[i] + "_cov", cov_log_returns, maturities=maturities, show_title=show_title)

            print("\n")


if __name__ == '__main__':
    analysis = Analysis()

    # pre-processing methods
    # analysis.normalisation_over_tenors()
    # analysis.standardisation_over_tenors()
    # analysis.logreturns_over_tenors()
    # analysis.normalisation_over_curves()

    # plot all the time-series
    analysis.all_data()






