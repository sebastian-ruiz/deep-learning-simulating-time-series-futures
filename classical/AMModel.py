"""
(simplified) Andersen Markov model module
"""
# pylint: skip-file
import copy
from multiprocessing import Pool
import scipy as sc
import scipy.linalg as scl
from scipy.interpolate import interp1d
import scipy.optimize
import scipy.linalg
import numpy as np
import numpy.matlib as ml
import logging

__all__ = ['AMModel']

class AMModel():
    """
    N factor Hull White model class
    """

    def __init__(self):
        """
        (simplified) Andersen Markov model class constructor.
        """
        self.n = 2
        self.eta1 = 1.0
        self.eta2 = 1.0
        self.eta_inf = 1.0
        self.kappa = 1.0
        self.alpha = 0.0
        self.rho = None
        self.cov = None
        self.eps = 0
        self.tenors = None
        self.obs_time = None
        self.delta = None
        self.delta_shift = None
        self.rates = None
        self.delta_t = 1 / 252
        self.shift = 1
        self.todayte = None
        self.make_parameters()

    def make_parameters(self):
        """
        Make AM model parameters

        :return:
        """
        self.rho = (self.eta1 / np.sqrt(self.eta1 * self.eta1 + self.eta2 * self.eta2)) * \
                   ((1 - np.exp(-self.kappa * self.delta_t)) / (self.kappa * np.sqrt(self.delta_t))) / \
                   np.sqrt((1 - np.exp(-2 * self.kappa * self.delta_t)) / (2 * self.kappa))
        
        
    def covariance(self, x=None):
        """
        This function gives the analytical covariance of the (simplified) Andersen Markov model

        :param eta1,eta2,etaInf,kappa: Andersen Markov model parameters
        :param tenors: historical tenors for the future curves
        """
        if x is not None:
            [eta1, eta2, eta_inf, kappa] = x
        else:
            eta1 = self.eta1
            eta2 = self.eta2
            eta_inf = self.eta_inf
            kappa = self.kappa
        cov_matrix = np.zeros((self.tenors.size, self.tenors.size))
        cov_z1_z1 = (eta1 * eta1 + eta2 * eta2) * (1 - np.exp(-2 * kappa * self.delta_t)) / (2 * kappa)
        cov_z1_z2 = eta1 * eta_inf * (1 - np.exp(- kappa * self.delta_t)) / kappa
        cov_z2_z2 = eta_inf * eta_inf * self.delta_t
        for i in range(0, self.tenors.size):
            for j in range(i, self.tenors.size):
                cov_matrix[i, j] = np.exp(- kappa * (self.tenors[i] + self.tenors[j])) * cov_z1_z1 + \
                                  (np.exp(- kappa * self.tenors[i]) + np.exp(- kappa * self.tenors[j])) *\
                                   cov_z1_z2 + cov_z2_z2
                cov_matrix[j, i] = cov_matrix[i, j]
        return cov_matrix

    def covariance_simplified(self, arg):
        """
        This function gives the analytical covariance of the (simplified) Andersen Markov model

        :param arg: vector of the parameters to be calibrated
        """
        [cov_z1_z1, cov_z1_z2, cov_z2_z2, kappa] = arg
        cov_matrix = np.zeros((self.tenors.size, self.tenors.size))
        for i in range(0, self.tenors.size):
            for j in range(i, self.tenors.size):
                cov_matrix[i, j] = np.exp(- kappa * (self.tenors[i] + self.tenors[j])) * cov_z1_z1 + \
                                  (np.exp(- kappa * self.tenors[i]) + np.exp(- kappa * self.tenors[j])) *\
                                   cov_z1_z2 + cov_z2_z2
                cov_matrix[j, i] = cov_matrix[i, j]
        return cov_matrix

    def make_data(self, m=10000):
        """
        This function can simulate the underlying process and rates provided AM Model parameters
        """
        
        if self.tenors is None:
            self.tenors = (np.array([i for i in range(1, 49)]))
        self.obs_time = np.array([i * self.delta_t for i in range(0, m)])
        self.delta = np.diff(self.obs_time)
        
        z = np.zeros((m, 2))
        rn = np.random.randn(m, 2)
        rnn = np.random.randn(m, self.tenors.size)

        # set the first new rate to the last one of the historical data
        last_rates = np.log(self.rates[-1, :])
        self.rates = np.zeros((m, self.tenors.size))
        self.rates[0, :] = last_rates
        for i in range(1, m):
            z[i,0] = np.exp(- self.kappa * self.delta_t) * z[i - 1, 0] + \
                   np.sqrt(self.eta1 * self.eta1 + self.eta2 * self.eta2) * \
                   np.sqrt((1 - np.exp(- 2 * self.kappa * self.delta_t)) / (2 * self.kappa)) * \
                   (self.rho * rn[i - 1, 0] + np.sqrt(1 - self.rho * self.rho) * rn[i - 1, 1])
            z[i, 1] = z[i - 1, 1] + self.eta_inf * np.sqrt(self.delta_t) * rn[i - 1, 0]
            c = (np.exp(- 2 * self.kappa * (self.tenors + self.obs_time[i])) * (np.exp(2 * self.kappa * self.obs_time[i]) - 1)
                 * (self.eta1 * self.eta1 + self.eta2 * self.eta2) + np.exp(- self.kappa * (self.tenors + self.obs_time[i])) *
                 (np.exp(self.kappa * self.obs_time[i]) - 1) * 4 * self.eta1 * self.eta_inf +
                 2 * self.obs_time[i] * self.kappa * self.eta_inf * self.eta_inf) / (4 * self.kappa)

            self.rates[i, :] = self.rates[0, :] \
                               + (np.exp(- self.kappa * self.tenors) * z[i, 0] + z[i, 1] - c).reshape((self.rates.shape[1],))

        self.obs_time = np.array([[i * self.delta_t for i in range(0, m)]])
        self.rates = np.exp(self.rates) # + rnn * self.eps
        self.data_sr = np.zeros((self.obs_time.size, self.n))
        self.delta = self.obs_time[1:, :] - self.obs_time[:-1, :]
        self.delta_shift = self.obs_time[self.shift:, :] - self.obs_time[:-self.shift, :]
        self.returns = np.log(self.rates[self.shift:, :] / self.rates[:-self.shift, :])
        self.cov = np.cov(self.returns.T)
        [self.data_sr, self.data_rn] = self.mlk([])

    def set_data(self, tenors, rates, obs_time, todayte):
        """
        Assigns historical future curve data to the object.

        :param tenors: rate tenors in year fractions
        :param rates: corresponding zero rates matrix
        :param obs_time: observation dates in year fractions (starting at the first date)
        """
        self.tenors = copy.deepcopy(tenors)
        self.obs_time = copy.deepcopy(obs_time)
        self.rates = copy.deepcopy(rates)
        self.data_sr = np.zeros((self.obs_time.size, self.n))
        self.delta = self.obs_time[1:, :] - self.obs_time[:-1, :]
        self.delta_shift = self.obs_time[self.shift:, :] - self.obs_time[:-self.shift, :]
        self.returns = np.log(self.rates[self.shift:, :] / self.rates[:-self.shift, :])
        self.cov = np.cov(self.returns.T)
        self.todayte = todayte

    def calibrate_moments(self, _=True, init=False):
        """
        This function calibrates class parameters to fit the historical data
        """
        return self.calibrate_moments_simplified(_, init)


    def calibrate_moments_simplified(self, _=True, init=False):
        """
        Calibrates model parameters to fit the historical data
        """
        res = sc.optimize.minimize(self.moments_target_fun_simplified,
                                   [self.cov[0, 0], self.cov[0, -1], self.cov[-1, -1], 0.5],
                                   args=(),
                                   method='Nelder-Mead',
                                   tol=None,
                                   callback=None,
                                   options={
                                       'disp': False,
                                       'initial_simplex': None, 'maxiter': None,
                                       'xatol': 1e-8, 'return_all': False,
                                       'fatol': 1e-8, 'maxfev': None})

        [cov_z1_z1, cov_z1_z2, cov_z2_z2, kappa] = res.x

        # Compute the model parameters
        eta_inf = np.sqrt(cov_z2_z2 / self.delta_t)
        eta1 = cov_z1_z2 / eta_inf * kappa / (1 - np.exp(-kappa * self.delta_t))
        eta2 = np.sqrt(cov_z1_z1 * 2 * kappa / (1 - np.exp(-2 * kappa * self.delta_t)) - eta1**2)

        # self.set_parameters('eta1', eta1)
        # self.set_parameters('eta2', eta2)
        # self.set_parameters('eta_inf', eta_inf)
        # self.set_parameters('kappa', kappa)
        self.make_parameters()
        v = [np.random.randn(self.obs_time.size, 2), np.random.randn(self.obs_time.size, 2)]
        self.optimality = self.mlk(ret=v)
        # self.data_sr = v[0]
        self.data_rn = v[1]

    def moments_target_fun_simplified(self, x):
        """
        Computes the difference between actuial covariance matrix and
        its approximation given by the model parameters
        """
        # find reasonable start values for the optimal search
        return np.sum(np.square(self.cov - self.covariance_simplified(x)))

    def mlk(self, ret=None):
        if ret is not None:
            alpha = 0
            t = 0
            x = np.zeros((2,self.obs_time.size))
            sigma1 = (np.exp(alpha * (t + self.tenors)) * (
                        self.eta1 * np.exp(- self.kappa * self.tenors) + self.eta_inf))
            sigma2 = (np.exp(alpha * (t + self.tenors)) * (self.eta2 * np.exp(- self.kappa * self.tenors)))
            b = np.concatenate((np.reshape(sigma1, (sigma1.size, 1)), np.reshape(sigma2, (sigma2.size, 1))), axis=1)
            mtmp = b.T @ b
            try:
                mtmp = scipy.linalg.inv(mtmp)
                b_l = mtmp @ b.T
                for i in range(1, self.obs_time.size):
                    x[:, i] = np.reshape(b_l @ self.returns[i-1:i,:].T,(2,))
            except Exception as e:
                logging.exception('Error calculating stochastic drivers for AMModel.')
            return [None, x.T]


    def is_valid(self):
        """
        If calibration accepted
        :return:
        """
        return self.kappa > 0.0 and self.valid
