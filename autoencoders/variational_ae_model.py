import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import metrics

from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
from config import Config

class VariationalAutoencoder():

    def __init__(self, params, plot=True):
        self.config = Config()
        self.plotting = Plotting()

        self.params = params
        self.plot = plot
        self.build_model(params)


    def build_model(self, params):
        input_dim = params['input_dim']
        latent_dim = params['latent_dim']

        x = Input(shape=(input_dim,))
        h = x
        for i in np.arange(len(params['hidden_layers'])):
            h = Dense(params['hidden_layers'][i])(h)
            h = LeakyReLU(alpha=params['leaky_relu'])(h)

        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=params['epsilon_std'])
            return z_mean + K.exp(z_log_var / 2) * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = z
        for i in np.flip(np.arange(len(params['hidden_layers']))):
            decoder_h = Dense(params['hidden_layers'][i], activation='relu')(decoder_h)
            decoder_h = LeakyReLU(alpha=params['leaky_relu'])(decoder_h)
        x_decoded_mean = Dense(input_dim, activation=params['last_activation'])(decoder_h)

        # instantiate VAE model
        self.vae = Model(x, x_decoded_mean)
        print("-" * 50, "\nvae")
        self.vae.summary()

        # build a model to project inputs on the latent space
        self.encoder = Model(x, z_mean)
        print("-" * 50, "\nencoder")
        self.encoder.summary()

        # build a digit generator that can sample from the learned distribution
        encoded_input = Input(shape=(latent_dim,))
        decoder_func = encoded_input
        for i in np.flip(np.arange(1, 2 * (len(params['hidden_layers']) + 1))):  # 2 * since we have dense layer and relu
            print("get layer:", -i)
            decoder_func = self.vae.layers[-i](decoder_func)

        self.generator_model = Model(encoded_input, decoder_func)
        print("-"*50, "\nencoder")
        self.generator_model.summary()

        def my_vae_loss(y_true, y_pred):
            if params['loss'] == 'binary_crossentropy':
                xent_loss = params['input_dim'] * metrics.binary_crossentropy(y_true, y_pred) # K.flatten(y_true), K.flatten(y_pred)
            else:
                xent_loss = params['input_dim'] * metrics.mean_squared_error(y_true, y_pred)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            vae_loss = K.mean(xent_loss + kl_loss)
            return vae_loss

        self.vae.compile(optimizer='adam', loss=my_vae_loss)

    def generator(self, x_train, batch_size):

        batch_train = np.zeros((batch_size, x_train.shape[1]))
        # batch_train_noisy = np.zeros((batch_size, x_train.shape[0], x_train.shape[1]))

        while True:
            for i in np.arange(batch_size):
                # choose random index in x_train
                index = np.random.choice((len(x_train)), 1)
                batch_train[i] = x_train[index]

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

        # checkpoint = ModelCheckpoint(self.config.get_filepath_ae_model("/checkpoints/vae-{epoch:02d}-{val_loss:.2f}"), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        # Train autoencoder for 50 epochs
        history = self.vae.fit_generator(self.generator(x_train, batch_size),
                                                 validation_data=(x_val, x_val),
                                                 steps_per_epoch=steps_per_epoch,
                                                 epochs=epochs,
                                                 verbose=2) # callbacks=[checkpoint],

        print(history.history.keys())

        if self.plot:
            plotting = Plotting()
            plotting.plot_loss(history.history['loss'], history.history['val_loss'], "deep_loss")

        if name is not None:
            self.save_model(name)

    def save_model(self, name):
        self.encoder.save(self.config.get_filepath_ae_model(name + "_encoder"))
        self.generator_model.save(self.config.get_filepath_ae_model(name + "_generator"))
        self.vae.save(self.config.get_filepath_ae_model(name + "_autoencoder"))

    def load_model(self, name):
        encoder_filepath = self.config.get_filepath_ae_model(name + "_encoder")
        generator_model_filepath = self.config.get_filepath_ae_model(name + "_generator")
        vae_filepath = self.config.get_filepath_ae_model(name + "_autoencoder")

        if self.config.file_exists(encoder_filepath) and self.config.file_exists(generator_model_filepath) and self.config.file_exists(vae_filepath):
            self.encoder = load_model(encoder_filepath, compile=False)
            self.generator_model = load_model(generator_model_filepath, compile=False)
            self.vae = load_model(vae_filepath, compile=False)
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
                temp.append(self.generator_model.predict(data[i]))
            return np.array(temp)

        else:
            return self.generator_model.predict(data)
