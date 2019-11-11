from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
from config import Config
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
# from pyramid.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from plotly.offline import plot, plot_mpl
import plotly.graph_objs as go

class Classical():

    def __init__(self):
        self.preprocess_data = PreprocessData()
        self.plotting = Plotting()
        self.config = Config()

        # self.preprocess_data.enable_min_max_scaler = True
        self.preprocess_data.enable_log_returns = True
        self.sets_training, self.sets_test, self.sets_training_scaled, self.sets_test_scaled, \
        self.training_dataset_names, self.test_dataset_names, self.maturities = self.preprocess_data.get_data()

        self.wti_nymex = self.sets_test[0]
        time = self.wti_nymex.axes[0].tolist()

        self.wti_nymex_short_end = self.wti_nymex.iloc[:, 0]
        self.data_scaled = self.sets_test_scaled[0][0]


        self.train_len = 128
        self.test_len = 42
        self.data_train = self.wti_nymex[:self.train_len]
        self.data_test = self.wti_nymex[self.train_len:self.train_len+self.test_len]
        self.data_train_and_test = self.wti_nymex[:self.train_len+self.test_len]

        print("self.data_train.shape", self.data_train.shape)
        print("self.data_test.shape", self.data_test.shape)

        # ACF and PACF plots

        # result = seasonal_decompose(wti_nymex_short_end, model='multiplicative')
        # fig2 = result.plot()
        # plot_mpl(fig2, image_filename="seasonal.html")

    def VAR(self):
        print("=" * 30 + "\nVAR\n" + "=" * 30 + "\n")
        # fit model
        model = VAR(self.data_train)
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.forecast(model_fit.y, steps=42)
        prediction = pd.DataFrame(yhat, index=self.data_test.index.values, columns=self.data_train.columns.values)

        curves = self.data_train_and_test

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)

        NUM_COLORS = curves.shape[1]
        cm = plt.get_cmap('coolwarm')
        ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)[::-1]])
        ax.plot(curves)
        custom_lines = [Line2D([0], [0], color=cm(1.), lw=4),
                        Line2D([0], [0], color=cm(0.), lw=4)]
        ax.legend(custom_lines, ['Short End', 'Long End'])
        plt.xticks(rotation=20)

        # plt.plot()
        plt.plot(prediction, color='red')

        plt.savefig(self.config.get_filepath_img("var"), dpi=300, bbox_inches='tight')
        plt.savefig(self.config.get_filepath_pgf("var"), dpi=300, transparent=True)  # , bbox_inches='tight'

        # plt.title("VAR")
        plt.show()

    def VMA(self):
        self.VARMA(order=(0, 1), name="VMA")

    def VARMA(self, order=(1,1), name="VARMA"):
        print("=" * 30 + "\n" + name + "\n" + "=" * 30 + "\n")
        # fit model
        model = VARMAX(self.data_train, order=order)
        model_fit = model.fit(disp=False)
        # make prediction
        yhat = model_fit.forecast(steps=42)
        prediction = pd.DataFrame(yhat, index=self.data_test.index.values, columns=self.data_train.columns.values)

        plt.plot(self.data_train_and_test)
        plt.plot(prediction, color='red')
        plt.title(name)
        plt.show()


    def AR(self):
        print("=" * 30 + "\nAR\n" + "=" * 30 + "\n")
        # fit model
        model = AR(self.data_train.iloc[:, 0])
        model_fit = model.fit()

        yhat = model_fit.predict(len(self.data_train), len(self.data_train)+42)
        # print(yhat)

        print(self.wti_nymex_short_end)

        # model_output = model_fit.fittedvalues
        plt.plot(self.wti_nymex_short_end[:84])
        plt.plot(yhat, color='red')
        # plt.title('AR RSS: %.4f' % np.nansum((model_output - self.data_scaled) ** 2))
        plt.show()

        # print(model_output.shape)
        print(self.data_scaled.shape)

        # df = pd.concat({'original':self.data_scaled, '0':model_output}, axis=1, sort=True)
        # df.fillna(0, inplace=True)
        # df.drop(['original'], axis=1, inplace=True)
        # df.rename({'0': 0}, axis='columns', inplace=True)
        #
        # print("df", df.head(30))
        #
        # rescaled = self.preprocess_data.rescale_data(df[0], self.test_dataset_names[0])
        #
        # plt.plot(rescaled)
        # plt.plot(self.sets_test[0][0])
        # plt.show()


    def MA(self):
        print("="*30 + "\nMA\n" + "="*30 + "\n")

        model = ARMA(self.data_scaled, order=(0, 1))
        model_fit = model.fit()
        # model_fit.summary()
        model_output = model_fit.fittedvalues
        plt.plot(self.data_scaled)
        plt.plot(model_output, color='red')
        plt.title('MA RSS: %.4f' % np.nansum((model_output - self.data_scaled) ** 2))
        plt.show()

        df = pd.concat({'original':self.data_scaled, '0':model_output}, axis=1, sort=True)
        df.fillna(0, inplace=True)
        df.drop(['original'], axis=1, inplace=True)
        df.rename({'0': 0}, axis='columns', inplace=True)

        rescaled = self.preprocess_data.rescale_data(df[0], self.test_dataset_names[0])

        plt.plot(rescaled)
        plt.plot(self.sets_test[0][0])
        plt.show()

    def ARMA(self):
        print("=" * 30 + "\nARMA\n" + "=" * 30 + "\n")
        model = ARMA(self.data_scaled, order=(2, 1))
        model_fit = model.fit()
        # model_fit.summary()
        model_output = model_fit.fittedvalues
        plt.plot(self.data_scaled)
        plt.plot(model_output, color='red')
        plt.title('ARMA RSS: %.4f' % np.nansum((model_output - self.data_scaled) ** 2))
        plt.show()

        df = pd.concat({'original':self.data_scaled, '0':model_output}, axis=1, sort=True)
        df.fillna(0, inplace=True)
        df.drop(['original'], axis=1, inplace=True)
        df.rename({'0': 0}, axis='columns', inplace=True)

        rescaled = self.preprocess_data.rescale_data(df[0], self.test_dataset_names[0])

        plt.plot(rescaled)
        plt.plot(self.sets_test[0][0])
        plt.show()


    def ARIMA(self):

        print("="*30 + "\nARIMA\n" + "="*30 + "\n")

        model = ARIMA(self.data_scaled, order=(1, 1, 1))
        model_fit = model.fit()
        # model_fit.summary()
        model_output = model_fit.fittedvalues
        plt.plot(self.data_scaled)
        plt.plot(model_output, color='red')
        plt.title('ARIMA RSS: %.4f' % np.nansum((model_output - self.data_scaled) ** 2))
        plt.show()

        df = pd.concat({'original':self.data_scaled, '0':model_output}, axis=1, sort=True)
        df.fillna(0, inplace=True)
        df.drop(['original'], axis=1, inplace=True)
        df.rename({'0': 0}, axis='columns', inplace=True)

        rescaled = self.preprocess_data.rescale_data(df[0], self.test_dataset_names[0])

        plt.plot(rescaled)
        plt.plot(self.sets_test[0][0])
        plt.show()



    def acf(self):
        sm.graphics.tsa.plot_acf(self.wti_nymex_short_end, lags=30)
        plt.show()


    def test_stationarity(self):
        timeseries = self.wti_nymex_short_end
        # Determing rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std() #pd.rolling_std(timeseries, window=12)

        # Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)

        # Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)


if __name__ == '__main__':
    classical = Classical()
    # classical.test_stationarity()

    classical.AR()
    # classical.ARMA()
    # classical.ARIMA()
    # # classical.acf()
    #
    # classical.VAR()
    # classical.VARMA()
    # classical.VMA()

