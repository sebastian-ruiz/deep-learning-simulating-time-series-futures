from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, MaxPooling1D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from helpers.plotting import Plotting
from config import Config
from autoencoders.adversarial_autoencoder_model import AdversarialAutoencoder
import numpy as np
import pandas as pd


class GAN:
    def __init__(self, params, plot=True):

        self.config = Config()
        self.plotting = Plotting()

        self.params = params
        self.plot = plot

        # Number of Conditioning, Random and Prediction returns
        self.num_c = params["num_c"]
        self.num_z = params["num_z"]
        self.num_o = params["num_o"]
        self.num_tenors = params["num_tenors"]

        optimizer = Adam(1e-5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=params["loss"],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        condition = Input(shape=(self.num_c, self.num_tenors))
        noise = Input(shape=(self.num_z, self.num_tenors))
        img = self.generator([condition, noise])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([condition, noise], validity)
        self.combined.compile(loss=params["loss"], optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        if self.params['gen_model_type'] == 'standard':
            model.add(Flatten(input_shape=(self.num_c + self.num_z, self.num_tenors)))

            for i in np.arange(len(self.params['gen_layers'])):
                model.add(Dense(self.params['gen_layers'][i], activation='relu')) # input_dim=(self.num_c + self.num_z, self.num_tenors)
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

    def build_discriminator(self):

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

    def train(self, data_train, name=None, sample_interval=200, epochs=None, batch_size=None):

        if epochs is None:
            epochs = self.params['epochs']
        if batch_size is None:
            batch_size = self.params['batch_size']

        discriminator_loss = []
        discriminator_acc = []
        generator_loss = []

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            real = self.collect_samples(data_train, 2*batch_size, self.num_c + self.num_o)
            real_labels = np.ones((2*batch_size, 1))

            d_loss_real = self.discriminator.train_on_batch(real, real_labels)

            # Generate a batch of new images
            condition = self.collect_samples(data_train, batch_size, self.num_c)
            noise = np.random.normal(size=(batch_size, self.num_z, self.num_tenors))  # THIS WORKS!
            gen_imgs = self.generator.predict([condition, noise])
            fake_labels = np.zeros((batch_size, 1))

            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake_labels)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            real = self.collect_samples(data_train, batch_size, self.num_c)  # THIS ALSO WORKS
            # noise = self.collect_samples(G, batch_size, num_z)  # THIS WORKS!
            noise = np.random.normal(size=(batch_size, self.num_z, self.num_tenors))
            real_labels = np.ones((batch_size, 1))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch([real, noise], real_labels)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # record progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
                discriminator_loss.append(d_loss[0])
                discriminator_acc.append(d_loss[1])
                generator_loss.append(g_loss)

                if np.isnan(d_loss[0]) or np.isnan(g_loss):
                    # something has gone wrong :(
                    break

                # plot simulation
                if self.plot:
                    generated, real_ = self.generate(condition=data_train, num_simulations=1)
                    self.plotting.plot_3d_training("gan_3d_simple_training/" + "%d" % epoch, generated, real_)

        if self.plot:
            self.plotting.plot_losses(discriminator_loss, discriminator_acc,generator_loss, "gan 3d simple training", legend = ['discriminator loss', 'discriminator acc', 'generator loss'])

        if name is not None:
            self.save_model(name)

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

        print("input predict shapes:", noise.shape, _condition.shape)

        generated = self.generator.predict([_condition, noise])

        if remove_condition:
            generated = generated[:, self.num_c:, :]

        if isinstance(repeat, int) and repeat > 0:
            for _ in np.arange(repeat - 1):
                generated_temp, _ = self.generate(condition=generated, remove_condition=True)
                generated = np.append(generated, generated_temp, axis=1)

        return generated, _condition

    def collect_samples(self, data, batch_size, pattern_len, ret_indices=False, indices=None):

        if type(data) is list:
            _data = np.array(data[np.random.randint(len(data))])
        else:
            _data = np.array(data)

        n = _data.shape[0] - pattern_len + 1
        if indices is None:
            indices = np.random.randint(n, size=batch_size)
        if ret_indices:
            return np.array([_data[a:a+pattern_len, :] for a in indices]), indices
        else:
            return np.array([_data[a:a+pattern_len, :] for a in indices])

    def save_model(self, name):
        self.generator.save(self.config.get_filepath_gan_model(name + "_3d_simple_generator"))
        self.discriminator.save(self.config.get_filepath_gan_model(name + "_3d_simple_discriminator"))
        self.combined.save(self.config.get_filepath_gan_model(name + "_3d_simple_combined"))

    def load_model(self, name):
        generator_filepath = self.config.get_filepath_gan_model(name + "_3d_simple_generator")
        discriminator_filepath = self.config.get_filepath_gan_model(name + "_3d_simple_discriminator")
        combined_filepath = self.config.get_filepath_gan_model(name + "_3d_simple_combined")

        if self.config.file_exists(generator_filepath) and self.config.file_exists(discriminator_filepath) and self.config.file_exists(combined_filepath):
            self.generator = load_model(generator_filepath)
            self.discriminator = load_model(discriminator_filepath)
            self.combined = load_model(combined_filepath)
            return True
        else:
            print("trained model does not exist yet!")
            print(self.config.file_exists(generator_filepath),self.config.file_exists(discriminator_filepath), self.config.file_exists(combined_filepath))
            print(generator_filepath, discriminator_filepath, combined_filepath)
            return False

    def load_else_train(self, x_train, name):
        did_load = self.load_model(name)
        if not did_load:
            self.train(x_train)
            self.save_model(name)