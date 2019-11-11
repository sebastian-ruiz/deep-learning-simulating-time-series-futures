from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten
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

class AutoencoderWindows():
    def __init__(self, params, plot=True):
        self.config = Config()
        self.plotting = Plotting()

        self.params = params
        self.plot = plot
        self.build_model(params)

    def build_model(self, params):
        input_dim = params['input_dim']
        latent_dim = params['latent_dim']
        # window = params['window']

        img_shape = input_dim

        # this is our input placeholder; 784 = 28 x 28
        input_data = Input(shape=img_shape)

        # "encoded" is the encoded representation of the inputs
        encoded = input_data
        encoded = Flatten()(encoded)
        for i in np.arange(len(params['hidden_layers'])):  # [56, 40, 28, 12, 4]
            encoded = Dense(params['hidden_layers'][i])(encoded)
            encoded = LeakyReLU(alpha=params['leaky_relu'])(encoded)

        encoded = Dense(latent_dim[0] * latent_dim[1])(encoded)
        encoded = Reshape(latent_dim)(encoded)
        encoded = LeakyReLU(alpha=0.1)(encoded)

        # "decoded" is the lossy reconstruction of the input
        decoded = encoded
        decoded = Flatten()(decoded)
        for i in np.flip(np.arange(len(params['hidden_layers']))):  # [56, 40, 28, 12, 4]
            decoded = Dense(params['hidden_layers'][i])(decoded)
            decoded = LeakyReLU(alpha=params['leaky_relu'])(decoded)

        decoded = Dense(img_shape[0] * img_shape[1], activation=params['last_activation'])(decoded)  # sigmoid or linear
        decoded = Reshape(img_shape)(decoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input_data, decoded)
        print("autoencoder")
        self.autoencoder.summary()

        # Separate Encoder model

        # this model maps an input to its encoded representation
        self.encoder = Model(input_data, encoded)
        print("encoder")
        self.encoder.summary()

        # Separate Decoder model

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=latent_dim)
        decoder_func = encoded_input
        for i in np.flip(np.arange(1, 2 * (len(params['hidden_layers']) + 3))):
            # 2 * since we have dense layer and relu
            # + 3 instead of +1 to account for the 2 reshapes
            decoder_func = self.autoencoder.layers[-i](decoder_func)

        self.decoder = Model(encoded_input, decoder_func)

        print("decoder")
        self.decoder.summary()

        # configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
        self.autoencoder.compile(optimizer='adam', loss=params['loss'])

    def generator(self, x_train, batch_size):

        input_dim = self.params['input_dim']

        if isinstance(input_dim, tuple) and len(input_dim) == 2:
            # train on windows of data instead
            window = input_dim[0]
            while True:

                if isinstance(x_train, list):
                    _data = np.array(x_train[np.random.randint(len(x_train))])
                else:
                    _data = np.array(x_train)

                n = _data.shape[0] - window + 1

                indices = np.random.randint(n, size=batch_size)
                batch_train = np.array([_data[a:a + window, :] for a in indices])

                # print("batch_train.shape", batch_train.shape, batch_size, window)

                yield batch_train, batch_train

        else:
            # train on individual curves

            if isinstance(x_train, list):
                _x_train = np.vstack(x_train)
            else:
                _x_train = x_train

            batch_train = np.zeros((batch_size, _x_train.shape[1]))
            # batch_train_noisy = np.zeros((batch_size, x_train.shape[0], x_train.shape[1]))

            while True:
                for i in np.arange(batch_size):
                    # choose random index in x_train
                    index = np.random.choice((len(_x_train)), 1)
                    batch_train[i] = _x_train[index]

                # add noise
                # batch_train_noisy = batch_train + np.random.normal(loc=0, scale=0.1, size=batch_train.shape)
                batch_train_noisy = batch_train
                # batch_train_noisy = np.clip(batch_train_noisy, 0., 1.)

                yield batch_train_noisy, batch_train

    def train(self, x_train, x_val, name=None, epochs=None, batch_size=None, steps_per_epoch=None):

        if epochs is None:
            epochs = self.params['epochs']
        if batch_size is None:
            batch_size = self.params['batch_size']
        if steps_per_epoch is None:
            steps_per_epoch = self.params['steps_per_epoch']

        # checkpoint = ModelCheckpoint(self.config.get_filepath_ae_model("/checkpoints/deep_ae_encoder-{epoch:02d}-{val_loss:.2f}"), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        # todo: make the x_val into a window of what we want.
        if isinstance(x_val, list):
            print("x_val", x_val[0].shape)

        # Train autoencoder for 50 epochs
        history = self.autoencoder.fit_generator(self.generator(x_train, batch_size),
                                                 # validation_data=(x_val, x_val),
                                                 steps_per_epoch=steps_per_epoch,
                                                 epochs=epochs,
                                                 verbose=2)# callbacks=[checkpoint],

        # history = self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True,
        #                           validation_data=(x_test, x_test), verbose=2)

        print(history.history.keys())

        if self.plot:
            plotting = Plotting()
            # plotting.plot_loss(history.history['loss'], history.history['val_loss'], "deep_loss")

        if name is not None:
            self.save_model(name)


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
                # temp.append(self.encoder.predict(data[i]))
                temp.append(self.encode(data[i]))
            return temp
        else:
            window_size = self.params['input_dim'][0]
            list_windows = self.chunks(np.array(data), window_size)

            # windows must all be the same size, sometimes cut off the end one.
            if list_windows[-1].shape[0] < window_size:
                windows = np.stack(list_windows[:-1])
            else:
                windows = np.stack(list_windows)

            return self.encoder.predict(windows) # as a test

    def chunks(self, l, n):
        n = max(1, n)
        return [l[i:i + n] for i in np.arange(0, len(l), n)]

    def decode(self, data):
        # if the data has FOUR dimensions then the first is the number of simulations
        # in the non-windowed case, three dimensions
        if len(data.shape) is 4:
            temp = []
            for i in np.arange(data.shape[0]):
                temp.append(self.decoder.predict(data[i]))
            return np.array(temp)

        else:
            # we probably want to merge the chunks together again after doing the decoding
            # we use np.vstack for that
            return np.vstack(self.decoder.predict(data))

    def evaluate(self, data, axis=None):

        encoded = self.encode(data)
        decoded = self.decode(encoded)

        mse = ((data - decoded) ** 2).mean(axis=axis)

        return mse

