from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model
import numpy as np
from helpers.plotting import Plotting
from config import Config
import matplotlib.pyplot as plt
from scipy.stats import norm


class AdversarialAutoencoder():
    def __init__(self, params, plot=True):

        self.config = Config()
        self.plotting = Plotting()
        self.params = params
        self.plot = plot

        self.input_dim = params['input_dim']
        self.latent_dim = params['latent_dim']

        optimizer = Adam(0.0002, 0.5)  # learning rate, beta_1

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(params)
        self.discriminator.compile(loss=params['loss_discriminator'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.build_encoder(params)
        self.decoder = self.build_decoder(params)

        img = Input(shape=(self.input_dim,))
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=[params['loss_generator'], params['loss_discriminator']],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)

    def build_encoder(self, params):
        # Encoder

        img = Input(shape=(self.input_dim,))
        h = img
        for i in np.arange(len(params['hidden_layers'])):
            h = Dense(params['hidden_layers'][i])(h)
            h = LeakyReLU(alpha=params['leaky_relu'])(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(params['latent_dim'])(h)
        latent_repr = Lambda(lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2), lambda p: p[0])(
            [mu, log_var])

        model = Model(img, latent_repr)
        print("-"*100, "\nencoder:")
        model.summary()
        return model

    def build_decoder(self, params):

        # model = Sequential()
        z = Input(shape=(params['latent_dim'],))
        h = z
        # for i in np.flip(np.arange(1, 2 * (len(params['hidden_layers']) + 1))):
        for i in np.flip(np.arange(len(params['hidden_layers']))):
            h = Dense(params['hidden_layers'][i])(h)
            h = LeakyReLU(alpha=params['leaky_relu'])(h)
        img = Dense(self.input_dim, activation=params['last_activation'])(h)

        model = Model(z, img)
        print("-"*100, "\ndecoder:")
        model.summary()
        return model

    def build_discriminator(self, params):

        # model = Sequential()
        encoded_repr = Input(shape=(self.latent_dim, ))
        h = encoded_repr
        h = Dense(self.latent_dim)(h)
        for i in np.arange(len(params['hidden_layers_discriminator'])):
            h = Dense(params['hidden_layers_discriminator'][i])(h)
            h = LeakyReLU(alpha=params['leaky_relu'])(h)
        validity = Dense(1, activation=self.params['last_activation_discriminator'])(h)

        model = Model(encoded_repr, validity)
        print("-"*100, "\ndiscriminator:")
        model.summary()
        return model

    def train(self, x_train, x_val, name=None, sample_interval=50, epochs=None, batch_size=None):

        if epochs is None:
            epochs = self.params['epochs']
        if batch_size is None:
            batch_size = self.params['batch_size']

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        discriminator_loss = []
        generator_loss = []
        generator_mse = []

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            latent_fake = self.encoder.predict(imgs)
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])

            # Plot the progress
            if sample_interval is not None and sample_interval != -1:
                if epoch % int(sample_interval) == 0:
                    print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # if epoch % sample_interval == 0:
                # self.plotting.plot_grid_1dim(self.config.get_filepath_img("/aae_training/" + str(epoch)), maturities, self.decoder)
            discriminator_loss.append(d_loss[0])
            generator_loss.append(g_loss[0])
            generator_mse.append(g_loss[1])

        print("[D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
        discriminator_loss[-1], 100 * discriminator_loss[-1], generator_loss[-1], generator_mse[-1]))

        if self.plot:
            self.plotting.plot_losses(discriminator_loss, generator_loss, generator_mse, "adversarial_losses")

        if name is not None:
            self.save_model(name)

    def save_model(self, name):
        self.encoder.save(self.config.get_filepath_ae_model(name + "_encoder"))
        self.decoder.save(self.config.get_filepath_ae_model(name + "_decoder"))
        self.discriminator.save(self.config.get_filepath_ae_model(name + "_discriminator"))

    def load_model(self, name):
        encoder_filepath = self.config.get_filepath_ae_model(name + "_encoder")
        decoder_filepath = self.config.get_filepath_ae_model(name + "_decoder")
        discriminator_filepath = self.config.get_filepath_ae_model(name + "_discriminator")

        if  self.config.file_exists(encoder_filepath) and  self.config.file_exists(decoder_filepath) and  self.config.file_exists(discriminator_filepath):
            self.encoder = load_model(encoder_filepath, compile=False)
            self.decoder = load_model(decoder_filepath, compile=False)
            self.discriminator = load_model(discriminator_filepath, compile=False)
            return True
        else:
            print("trained model does not exist yet!")
            return False

    def load_else_train(self, x_train, x_val, name):
        did_load = self.load_model(name)
        if not did_load:
            self.train(x_train, x_val)
            self.save_model(name)

    def encode(self, data):
        # if the data is a list then encode each item separately
        if isinstance(data, list):
            temp = []
            for i in np.arange(len(data)):
                temp.append(self.encoder.predict(data[i]))
            return temp
        else:
            return self.encoder.predict(data)

    def decode(self, data):
        # if the data has three dimensions then the first is the number of simulations
        if len(data.shape) is 3:
            temp = []
            for i in np.arange(data.shape[0]):
                temp.append(self.decoder.predict(data[i]))
            return np.array(temp)

        else:
            return self.decoder.predict(data)
