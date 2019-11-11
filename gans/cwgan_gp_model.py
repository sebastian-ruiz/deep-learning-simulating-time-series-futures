# Large amount of credit goes to:
# https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py and
# https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py
# which I've used as a reference for this implementation
# Author: Hanling Wang
# Date: 2018-11-21

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, MaxPooling1D
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop
from functools import partial
from config import Config
from helpers.plotting import Plotting
import keras.backend as K
import math
import numpy as np
import pandas as pd

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def __init__(self, batch_size):
        super(RandomWeightedAverage, self).__init__()
        self.batch_size = batch_size

    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class CWGANGP():
    def __init__(self, params, plot=True):
        self.config = Config()
        self.plotting = Plotting()

        self.params = params
        self.plot = plot
        self.num_c = 6*7
        self.num_o = 6*7
        self.num_z = 6*7
        self.num_tenors = params['num_tenors']
        self.losslog = []
        self.epochs = params['epochs'] # 100
        self.batch_size = params['batch_size'] # 32
        self.sample_interval = params['sample_interval'] # 50

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=(self.num_c + self.num_o, self.num_tenors))

        # Noise input
        z_disc = Input(shape=(self.num_z, self.num_tenors))

        # Generate image based of noise (fake sample) and add label to the input
        condition = Input(shape=(self.num_c, self.num_tenors))
        fake_img = self.generator([condition, z_disc])

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage(self.batch_size)([real_img, fake_img])

        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_img, condition, z_disc], outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, 10])
        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.num_z, self.num_tenors))
        # add label to the input
        condition = Input(shape=(self.num_c, self.num_tenors))
        # Generate images based of noise
        img = self.generator([condition, z_gen])
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model([condition, z_gen], valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        if self.params['gen_model_type'] == 'standard':
            model.add(Flatten(input_shape=(self.num_c + self.num_z, self.num_tenors)))

            for i in np.arange(len(self.params['gen_layers'])):
                model.add(Dense(self.params['gen_layers'][i], activation='relu'))
                # model.add(LeakyReLU(alpha=self.params['leaky_relu']))

        elif self.params['gen_model_type'] == 'conv':
            model.add(Conv1D(28, kernel_size=5, padding="same", data_format="channels_last",
                             activation='relu', input_shape=(self.num_c + self.num_z, self.num_tenors)))  # for termporal data we should use padding valid
            model.add(Conv1D(2, kernel_size=3, padding="same", data_format="channels_last",
                             activation='relu', input_shape=(self.num_c + self.num_z, self.num_tenors)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())

        # final layers
        model.add(Dense(np.prod((self.num_o, self.num_tenors)), activation=self.params['gen_last_activation']))
        model.add(Reshape((self.num_o, self.num_tenors)))

        print("-"*20 + "\ngan generator")
        model.summary()

        condition = Input(shape=(self.num_c, self.num_tenors))
        z = Input(shape=(self.num_z, self.num_tenors))
        model_input = concatenate([condition, z], axis=1)

        out = model(model_input)

        return Model([condition, z], concatenate([condition, out], axis=1))

    def build_critic(self):

        model = Sequential()

        if self.params['dis_model_type'] == 'standard':
            model.add(Flatten(input_shape=(self.num_c + self.num_o, self.num_tenors)))

            for i in np.arange(len(self.params['dis_layers'])):
                model.add(Dense(self.params['dis_layers'][i], activation='relu'))
                # model.add(LeakyReLU(alpha=self.params['leaky_relu']))

        elif self.params['dis_model_type'] == 'conv':
            model.add(Conv1D(32, kernel_size=4, strides=1, padding='same', activation='relu',
                             input_shape=(self.num_c + self.num_z, self.num_tenors)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())


        # final layer
        model.add(Dense(1, activation=self.params['dis_last_activation']))

        print("-" * 20 + "\ngan discriminator")
        model.summary()

        model_input = Input(shape=(self.num_c + self.num_o, self.num_tenors))
        validity = model(model_input)

        return Model(model_input, validity)

    def train(self, data, name=None):

        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))  # Dummy gt for gradient penalty
        for epoch in range(self.epochs):
            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                condition, real = self.collect_samples(data, self.batch_size, self.num_o, self.num_z)

                # Sample generator input
                noise = np.random.normal(size=(self.batch_size, self.num_z, self.num_tenors)) # this was used for the GAN
                # Train the critic
                cond_real = np.concatenate([condition, real], axis=1)
                d_loss = self.critic_model.train_on_batch([cond_real, condition, noise], [valid, fake, dummy]) # real_img, condition, z_disc

            # ---------------------
            #  Train Generator
            # ---------------------
            condition = self.collect_samples(data, self.batch_size, self.num_c)
            noise = np.random.normal(size=(self.batch_size, self.num_z, self.num_tenors))
            g_loss = self.generator_model.train_on_batch([condition, noise], valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            self.losslog.append([d_loss[0], g_loss])

            if np.isnan(d_loss[0]) or np.isnan(g_loss):
                # something has gone wrong :(
                break

            # If at save interval => save generated image samples
            # if epoch % self.sample_interval == 0:
            #     generated, real_ = self.generate(data=data, num_simulations=100, random_data=True)
            #     self.plotting.plot_3d_training("gan_3d_simple_training/" + "%d" % epoch, generated[0], real_)
            #     with open('loss.log', 'w') as f:
            #         f.writelines('d_loss, g_loss\n')
            #         for each in self.losslog:
            #             f.writelines('%s, %s\n' % (each[0], each[1]))

        if name is not None:
            self.save_model(name)

    def collect_samples(self, data, batch_size, real_len, condition_len=0):

        if type(data) is list:
            _data = np.array(data[np.random.randint(len(data))])
        else:
            _data = np.array(data)

        total_len = real_len + condition_len

        n = _data.shape[0] - total_len + 1
        indices = np.random.randint(n, size=batch_size)
        if condition_len > 0:
            condition = np.array([_data[a:a+condition_len, :] for a in indices])
            real = np.array([_data[a+condition_len:a+total_len , :] for a in indices])
            return condition, real
        else:
            return np.array([_data[a:a+total_len, :] for a in indices])

    def generate(self, condition=None, condition_on_end=True, num_simulations=1, remove_condition=True, repeat=None):

        if isinstance(condition, pd.DataFrame):
            _condition = np.array(condition)
        else:
            _condition = condition.copy()

        print("_condition", _condition.shape)

        if condition_on_end:
            if isinstance(condition, list):
                _condition = _condition[0][np.newaxis, -self.num_c:]
            elif len(condition.shape) == 2:
                _condition = _condition[np.newaxis, -self.num_c:]
            else:
                _condition = _condition[:, -self.num_c:]
        else: # not condition_on_end:
            if type(condition) is list:
                _condition = _condition[0][np.newaxis, :self.num_c]
            elif len(condition.shape) == 2:
                _condition = _condition[np.newaxis, :self.num_c]
            else: # len(condition.shape) == 3:
                _condition = _condition[:, :self.num_c]

        print("_condition after", _condition.shape)

        # override num_simulations if _conditions already is a 2d array
        _num_simulations = 1
        if num_simulations > 1:
            _condition = np.repeat(_condition, num_simulations, axis=0)
            _num_simulations = num_simulations
        elif len(_condition.shape) > 1 and _condition.shape[0] is not 1:
            _num_simulations = _condition.shape[0]

        noise = np.random.normal(size=(_num_simulations, self.num_z, self.num_tenors))
        generated = self.generator.predict([_condition, noise])

        if remove_condition:
            generated = generated[:, self.num_c:, :]

        if isinstance(repeat, int) and repeat > 0:
            for _ in np.arange(repeat - 1):
                generated_temp, _ = self.generate(condition=generated, remove_condition=True)
                generated = np.append(generated, generated_temp, axis=1)

        return generated, _condition


    def save_model(self, name):
        self.generator.save(self.config.get_filepath_gan_model(name + "_cwgan_gp_generator"))
        self.critic.save(self.config.get_filepath_gan_model(name + "_cwgan_gp_critic"))
        # self.combined.save(self.config.get_filepath_gan_model(name + "_3d_simple_combined"))

    def load_model(self, name):
        generator_filepath = self.config.get_filepath_gan_model(name + "_cwgan_gp_generator")
        critic_filepath = self.config.get_filepath_gan_model(name + "_cwgan_gp_critic")
        # combined_filepath = self.config.get_filepath_ae_model(name + "_3d_simple_combined")

        if self.config.file_exists(generator_filepath) and self.config.file_exists(critic_filepath):
            self.generator = load_model(generator_filepath)
            self.critic = load_model(critic_filepath)
            # self.combined = load_model(combined_filepath)
            return True
        else:
            print("trained model does not exist yet!")
            return False

    def load_else_train(self, x_train, name):
        did_load = self.load_model(name)
        if not did_load:
            self.train(x_train)
            self.save_model(name)