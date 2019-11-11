from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.models import load_model
import numpy as np
from helpers.plotting import Plotting
from config import Config
import matplotlib.pyplot as plt
from scipy.stats import norm
from helpers.evaluate import reconstruction_error


class BaseAutoencoder:
    def __init__(self, params):
        self.config = Config()
        self.plotting = Plotting()

        self.params = params
        # self.build_model(params)

    def train(self, x_train, x_val, name=None, epochs=None, batch_size=None, steps_per_epoch=None):
        print("not implemented")


    def save_model(self, name):
        self.encoder.save(self.config.get_filepath_ae_model(name + "_encoder"))
        self.decoder.save(self.config.get_filepath_ae_model(name + "_decoder"))
        self.autoencoder.save(self.config.get_filepath_ae_model(name + "_autoencoder"))

    def load_model(self, name):
        encoder_filepath = self.config.get_filepath_ae_model(name + "_encoder")
        decoder_filepath = self.config.get_filepath_ae_model(name + "_decoder")
        autoencoder_filepath = self.config.get_filepath_ae_model(name + "_autoencoder")

        if self.config.file_exists(encoder_filepath) and self.config.file_exists(decoder_filepath) and self.config.file_exists(autoencoder_filepath):
            self.encoder = load_model(encoder_filepath)
            self.decoder = load_model(decoder_filepath)
            self.autoencoder = load_model(autoencoder_filepath)
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

    def evaluate(self, data, axis=None):

        encoded = self.encode(data)
        decoded = self.decode(encoded)

        mse = ((data - decoded) ** 2).mean(axis=axis)

        return mse