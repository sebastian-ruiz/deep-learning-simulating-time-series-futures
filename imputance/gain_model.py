import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tqdm import tqdm
from helpers.plotting import Plotting
from config import Config
import pandas as pd
import math

class gain():
    def __init__(self, params):
        self.config = Config()
        self.plotting = Plotting()
        self.params = params


        # %% System Parameters
        # 1. Mini batch size
        self.mb_size = params['mb_size']
        # 2. Missing rate
        self.p_miss = params['p_miss']
        # 3. Hint rate
        self.p_hint = params['p_hint']
        # 4. Loss Hyperparameters
        self.alpha = params['alpha']
        # 5. Input Dim (Fixed)
        self.dim = params['dim']
        # 6. No
        self.train_no = params['train_no']
        self.test_no = params['test_no']

        # Sessions
        self.sess = tf.Session()


    # Mask Vector and Hint Vector Generation
    def sample_M(self, m, n, p):
        A = np.random.uniform(0., 1., size=[m, n])
        B = A > p
        C = 1. * B
        return C

    # %% Necessary Functions
    # 1. Xavier Initialization Definition
    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)


    # 2. Plot (4 x 4 subfigures)
    def plot(self, samples):
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        return fig

    # %% 3. Others
    # Random sample generator for Z
    def sample_Z(self, m, n):
        return np.random.uniform(0., 1., size=[m, n])

    def sample_idx(self, m, n):
        A = np.random.permutation(m)
        idx = A[:n]
        return idx

    def make_mask(self, data):

        mask = pd.DataFrame(data=np.ones(data.shape), index=data.index, columns=data.columns)
        mask.mask(np.random.random(data.shape) < .05, inplace=True)
        mask.fillna(0, inplace=True)

        random_indexes = np.random.random(mask.shape) < .05
        for i in np.arange(len(mask)):
            if random_indexes[i]:
                random_size_interval = np.random.randint(10, 20)
                mask.iloc[i: i + random_size_interval] = 0

        return mask


    def build_model(self):
        '''
        GAIN Consists of 3 Components
        - Generator
        - Discriminator
        - Hint Mechanism
        '''

        # %% GAIN Architecture

        # %% 1. Input Placeholders
        # 1.1. Data Vector
        self.X = tf.placeholder(tf.float32, shape=[None, self.dim])
        # 1.2. Mask Vector
        self.M = tf.placeholder(tf.float32, shape=[None, self.dim])
        # 1.3. Hint vector
        self.H = tf.placeholder(tf.float32, shape=[None, self.dim])
        # 1.4. Random Noise Vector
        self.Z = tf.placeholder(tf.float32, shape=[None, self.dim])

        # %% 2. Discriminator
        D_W1 = tf.Variable(self.xavier_init([self.dim * 2, 32]))  # Data + Hint as inputs
        D_b1 = tf.Variable(tf.zeros(shape=[32]))

        D_W2 = tf.Variable(self.xavier_init([32, 16]))
        D_b2 = tf.Variable(tf.zeros(shape=[16]))

        D_W3 = tf.Variable(self.xavier_init([16, self.dim]))
        D_b3 = tf.Variable(tf.zeros(shape=[self.dim]))  # Output is multi-variate

        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

        # %% 3. Generator
        G_W1 = tf.Variable(self.xavier_init([self.dim * 2, 32]))  # Data + Mask as inputs (Random Noises are in Missing Components)
        G_b1 = tf.Variable(tf.zeros(shape=[32]))

        G_W2 = tf.Variable(self.xavier_init([32, 16]))
        G_b2 = tf.Variable(tf.zeros(shape=[16]))

        G_W3 = tf.Variable(self.xavier_init([16, self.dim]))
        G_b3 = tf.Variable(tf.zeros(shape=[self.dim]))

        theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

        # %% GAIN Function

        # %% 1. Generator
        def generator(x, z, m):
            inp = m * x + (1 - m) * z  # Fill in random noise on the missing values
            inputs = tf.concat(axis=1, values=[inp, m])  # Mask + Data Concatenate
            G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
            G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
            G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)  # [0,1] normalized Output

            return G_prob

        # %% 2. Discriminator
        def discriminator(x, m, g, h):
            inp = m * x + (1 - m) * g  # Replace missing values to the imputed values
            inputs = tf.concat(axis=1, values=[inp, h])  # Hint + Data Concatenate
            D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
            D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
            D_logit = tf.matmul(D_h2, D_W3) + D_b3
            D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output

            return D_prob



        # %% Structure
        self.G_sample = generator(self.X, self.Z, self.M)
        D_prob = discriminator(self.X, self.M, self.G_sample, self.H)

        # %% Loss
        self.D_loss1 = -tf.reduce_mean(self.M * tf.log(D_prob + 1e-8) + (1 - self.M) * tf.log(1. - D_prob + 1e-8)) * 2
        self.G_loss1 = -tf.reduce_mean((1 - self.M) * tf.log(D_prob + 1e-8)) / tf.reduce_mean(1 - self.M)
        self.MSE_train_loss = tf.reduce_mean((self.M * self.X - self.M * self.G_sample) ** 2) / tf.reduce_mean(self.M)

        D_loss = self.D_loss1
        G_loss = self.G_loss1 + self.alpha * self.MSE_train_loss

        # %% MSE Performance metric
        self.MSE_test_loss = tf.reduce_mean(((1 - self.M) * self.X - (1 - self.M) * self.G_sample) ** 2) / tf.reduce_mean(1 - self.M)

        # %% Solver
        self.D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
        self.G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    def train(self, x_train, x_test, test_mask=None):
        # session
        self.sess.run(tf.global_variables_initializer())

        _x_test = np.array(x_test)

        if test_mask is None:
            test_m = np.array(self.make_mask(x_test))  # np.array(test_mask)
        else:
            test_m = np.array(test_mask)

        # %%
        # Output Initialization
        if not os.path.exists('Multiple_Impute_out1/'):
            os.makedirs('Multiple_Impute_out1/')

        # Iteration Initialization
        i = 1

        # %% Start Iterations
        for it in tqdm(range(self.params['epochs'])):

            if type(x_train) is list:
                rand_index = np.random.randint(len(x_train))
                _train = np.array(x_train[rand_index])
                train_m = np.array(self.make_mask(x_train[rand_index]))
            else:
                _train = np.array(x_train)
                train_m = np.array(self.make_mask(x_train))


            # %% Inputs
            mb_idx = self.sample_idx(self.train_no, self.mb_size)
            X_mb = _train[mb_idx, :]
            Z_mb = self.sample_Z(self.mb_size, self.dim)
            M_mb = train_m[mb_idx, :]
            H_mb1 = self.sample_M(self.mb_size, self.dim, 1 - self.p_hint)
            H_mb = M_mb * H_mb1

            New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

            _, D_loss_curr = self.sess.run([self.D_solver, self.D_loss1], feed_dict={self.X: X_mb, self.M: M_mb, self.Z: New_X_mb, self.H: H_mb})
            _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = self.sess.run(
                [self.G_solver, self.G_loss1, self.MSE_train_loss, self.MSE_test_loss],
                feed_dict={self.X: X_mb, self.M: M_mb, self.Z: New_X_mb, self.H: H_mb})

            # %% Output figure
            if it % 100 == 0:
                mb_idx = self.sample_idx(self.test_no, 5)
                X_mb = _x_test[mb_idx, :]
                M_mb = test_m[mb_idx, :]
                Z_mb = self.sample_Z(5, self.dim)

                New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

                samples1 = X_mb
                samples5 = M_mb * X_mb + (1 - M_mb) * Z_mb

                samples2 = self.sess.run(self.G_sample, feed_dict={self.X: X_mb, self.M: M_mb, self.Z: New_X_mb})
                samples2 = M_mb * X_mb + (1 - M_mb) * samples2

                Z_mb = self.sample_Z(5, self.dim)
                New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
                samples3 = self.sess.run(self.G_sample, feed_dict={self.X: X_mb, self.M: M_mb, self.Z: New_X_mb})
                samples3 = M_mb * X_mb + (1 - M_mb) * samples3

                Z_mb = self.sample_Z(5, self.dim)
                New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
                samples4 = self.sess.run(self.G_sample, feed_dict={self.X: X_mb, self.M: M_mb, self.Z: New_X_mb})
                samples4 = M_mb * X_mb + (1 - M_mb) * samples4

                samples = np.vstack([samples5, samples2, samples3, samples4, samples1])
                # print(samples)
                # fig = self.plot(samples)
                # plt.savefig('Multiple_Impute_out1/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                # i += 1
                # plt.close(fig)

            if math.isnan(MSE_train_loss_curr) or math.isnan(MSE_train_loss_curr):
                print("training failed due to NaN")
                break

            # %% Intermediate Losses
            if it % 100 == 0:
                print('Iter: {}'.format(it))
                print('Train_loss: {:.4}'.format(MSE_train_loss_curr))
                print('Test_loss: {:.4}'.format(MSE_test_loss_curr))
                print()

    def predict(self, data, mask):

        X_mb = np.array(data)
        Z_mb = self.sample_Z(X_mb.shape[0], self.dim)
        M_mb = np.array(mask)

        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

        output = self.sess.run(self.G_sample, feed_dict={self.X: X_mb, self.M: M_mb, self.Z: New_X_mb})
        output = M_mb * X_mb + (1 - M_mb) * output

        # return output

        return pd.DataFrame(data=output, index=data.index, columns=data.columns)
