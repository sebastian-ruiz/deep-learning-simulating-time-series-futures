import numpy as np
import matplotlib.pyplot as plt
from config import Config
from helpers.plotting import Plotting

class PCAModel():
    def __init__(self, params, plot=True):

        self.k = params['latent_dim']
        self.A_tilde = None
        self.mu = None
        self.plot = plot

        self.config = Config()
        self.plotting = Plotting()

        print("PCA")

    def train(self, X, X_val=None, name=None):

        # set each column to be a sample realisation
        X = X.T

        Q = np.cov(X)
        L, A = np.linalg.eig(Q)
        # sort eigenvalues from largest to smallest
        idx = L.argsort()[::-1]
        L = L[idx]
        A = A[:, idx]

        self.mu = np.mean(X, axis=1)
        Y = A.T @ (X - self.mu.reshape(len(self.mu), 1))

        print(X.shape, self.mu.shape, Q.shape, L.shape, A.shape, Y.shape)

        variability = [np.sum(L[:k])/np.sum(L) for k in np.arange(len(L))]
        print(variability)

        self.A_tilde = A[:, :self.k]

        X_approx = self.mu.reshape(len(self.mu), 1) + A[:, :self.k] @ Y[:self.k, :]

        if self.plot:
            # plot first three loadings
            plt.figure(figsize=(4, 3), dpi=300)
            maturities = np.arange(X.shape[0]) * 1/12 # each tenor is 30 days, divide by 12 for years
            line_1, = plt.plot(maturities, A[:, 0], label="$a_1$")
            line_2, = plt.plot(maturities, A[:, 1], label="$a_2$")
            line_3, = plt.plot(maturities, A[:, 2], label="$a_3$")
            plt.legend(handles=[line_1, line_2, line_3])
            plt.grid(True)
            plt.xlabel('Term to maturity (years)')
            plt.ylabel('Loading values')
            file_name = "pca_loadings"
            plt.savefig(self.config.get_filepath_img(file_name), dpi=300, bbox_inches='tight')
            plt.savefig(self.config.get_filepath_pgf(file_name), dpi=300, transparent=True)
            plt.show()

    def save_model(self, name):
        print("save PCA model, not implemented")

    def load_model(self, name):
        print("load PCA model, not implemented")

    def load_else_train(self, x_train, x_val, name):
        self.train(x_train, x_val)

    def encode(self, X_hat): # X_hat
        # if the data is a list then encode each item separately
        if isinstance(X_hat, list):
            temp = []
            for i in np.arange(len(X_hat)):
                temp.append(self.encode(X_hat[i]))
            return temp
        else:
            _X_hat = np.array(X_hat)
            _X_hat_T = _X_hat.T

            mu_hat = np.mean(_X_hat_T, axis=1)
            Y_hat = self.A_tilde.T @ (_X_hat_T - mu_hat.reshape(len(mu_hat), 1))

            return Y_hat.T

    def decode(self, Y_hat_transpose):

        if len(Y_hat_transpose.shape) is 3:
            temp = []
            for i in np.arange(Y_hat_transpose.shape[0]):
                temp.append(self.decode(Y_hat_transpose[i]))
            return np.array(temp)

        else:
            Y_hat = np.array(Y_hat_transpose).T
            X_hat_approx = self.mu.reshape(len(self.mu), 1) + self.A_tilde @ Y_hat

            return X_hat_approx.T