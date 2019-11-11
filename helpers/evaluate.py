from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from enum import Enum


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self) :
        for f in self.files:
            f.flush()


class PreprocessType(Enum):
    NONE = 0
    NORMALISATION_OVER_TENORS  = 1
    STANDARDISATION_OVER_TENORS = 2
    LOG_RETURNS_OVER_TENORS = 3
    NORMALISATION_OVER_CURVES = 4
    STANDARDISATION_OVER_CURVES = 5
    LOG_RETURNS_OVER_CURVES = 6


class AEModel(Enum):
    AE  = 1
    AAE = 2
    VAE = 3
    PCA = 4
    AE_WINDOWS = 5


class GANModel(Enum):
    GAN = 1
    WGAN = 2
    GAN_CONV = 3


def reconstruction_error(input, output):
    # mse = mean_squared_error(input, output)
    mse = ((input - output) ** 2).mean(axis=1)
    mse = np.around(mse, 3)
    return mse


def log_returns_over_tenors(df):
    if isinstance(df, pd.DataFrame):
        _df = df
    else:
        _df = pd.DataFrame(df)

    # we calculate log(F(t_i, T)/F(t_{i-1}, T)), for T=T_1,..., T_n
    shift = (_df.shift(0)) / (_df.shift(1))
    shift = shift.dropna()
    return np.array(np.log(shift))


def cov_log_returns_over_tenors(df):
    # we transpose to get an (n x n) matrix instead of an ((m-1) x (m-1)) matrix
    # where m is the number of dates, and n the number of tenors
    return np.cov(np.transpose(log_returns_over_tenors(df)))


def smape(A, F, over_curves=False):  # A is simulated, F is real

    if isinstance(A, pd.DataFrame):
        _A = np.array(A)
    else:
        _A = A

    if isinstance(F, pd.DataFrame):
        _F = np.array(F)
    else:
        _F = F

    if (isinstance(_A, np.ndarray) and len(_A.shape) == 3) or over_curves:
        smape_results = []
        for i in np.arange(_A.shape[0]):
            if _A.shape == _F.shape:
                smape_results.append(smape(_A[i], _F[i]))
            else:
                smape_results.append(smape(_A[i], _F))
        return smape_results
    else:
        if _A.shape != _F.shape:
            print("(smape) shapes are not the same size!")
            raise SystemExit(0)

        n = np.prod(A.shape)  # number of terms that we are summing over
        return 1/n * np.sum(2 * np.abs(_F - _A) / (np.abs(_A) + np.abs(_F)))
