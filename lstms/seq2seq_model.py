import numpy as np
import keras
from time import time
from lstms.utils import random_sine, plot_prediction
from pathlib import Path
import os

class Model():

    def __init__(self, layers, learning_rate, decay, num_input_features,
                 num_output_features, loss, lambda_regulariser, regulariser,
                 batch_size, steps_per_epoch, epochs, input_sequence_length,
                 target_sequence_length, num_steps_to_predict):

        self.optimiser = keras.optimizers.Adam(lr=learning_rate, decay=decay)
        self.layers = layers
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.loss = loss
        self.lambda_regulariser = lambda_regulariser
        self.regulariser = regulariser
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length
        self.num_steps_to_predict = num_steps_to_predict

        self.num_signals = 10  # The number of random sine waves the compose the signal. The more sine waves, the harder the problem.

        models_path = str(Path(__file__).parent.resolve()) + '/saved_model'
        self.logs_path = str(Path(__file__).parent.resolve()) + '/logs'
        self.saved_model_path = models_path + '/model.h5'
        self.saved_decoder_model_path = models_path + '/model_decoder.h5'
        self.saved_encoder_model_path = models_path + '/model_encoder.h5'

        self.model = None
        self.encoder_predict_model = None
        self.decoder_predict_model = None

        # self.data_generator = DataGenerator()

    def load(self):
        if os.path.isfile(self.saved_encoder_model_path) \
                and os.path.isfile(self.saved_decoder_model_path)\
                and os.path.isfile(self.saved_model_path):

            self.model = keras.models.load_model(self.saved_model_path)
            self.encoder_predict_model = keras.models.load_model(self.saved_encoder_model_path)
            self.decoder_predict_model = keras.models.load_model(self.saved_decoder_model_path)

    def _save(self):
        self.model.save(self.saved_model_path)
        self.encoder_predict_model.save(self.saved_encoder_model_path)
        self.decoder_predict_model.save(self.saved_decoder_model_path)

    def build(self):

        # Define an input sequence.
        encoder_inputs = keras.layers.Input(shape=(None, self.num_input_features))

        # Create a list of RNN Cells, these are then concatenated into a single layer
        # with the RNN layer.
        encoder_cells = []
        for hidden_neurons in self.layers:
            encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                      kernel_regularizer=self.regulariser,
                                                      recurrent_regularizer=self.regulariser,
                                                      bias_regularizer=self.regulariser))

        encoder = keras.layers.RNN(encoder_cells, return_state=True)

        encoder_outputs_and_states = encoder(encoder_inputs)

        # Discard encoder outputs and only keep the states.
        # The outputs are of no interest to us, the encoder's
        # job is to create a state describing the input sequence.
        encoder_states = encoder_outputs_and_states[1:]

        # NEXT

        # The decoder input will be set to zero (see random_sine function of the utils module).
        # Do not worry about the input size being 1, I will explain that in the next cell.
        decoder_inputs = keras.layers.Input(shape=(None, 1))

        decoder_cells = []
        for hidden_neurons in self.layers:
            decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                      kernel_regularizer=self.regulariser,
                                                      recurrent_regularizer=self.regulariser,
                                                      bias_regularizer=self.regulariser))

        decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

        # Set the initial state of the decoder to be the output state of the encoder.
        # This is the fundamental part of the encoder-decoder.
        decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

        # Only select the output of the decoder (not the states)
        decoder_outputs = decoder_outputs_and_states[0]

        # Apply a dense layer with linear activation to set output to correct dimension
        # and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
        decoder_dense = keras.layers.Dense(self.num_output_features,
                                           activation='linear',
                                           kernel_regularizer=self.regulariser,
                                           bias_regularizer=self.regulariser)

        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        self.model.compile(optimizer=self.optimiser, loss=self.loss)
        self.model.summary()


        # todo: does the above model need to be trained first?
        # The following is for the more complex model

        self.encoder_predict_model = keras.models.Model(encoder_inputs, encoder_states)

        decoder_states_inputs = []

        # Read layers backwards to fit the format of initial_state
        # For some reason, the states of the model are order backwards (state of the first layer at the end of the list)
        # If instead of a GRU you were using an LSTM Cell, you would have to append two Input tensors since the LSTM has 2 states.
        for hidden_neurons in self.layers[::-1]:
            # One state for GRU
            decoder_states_inputs.append(keras.layers.Input(shape=(hidden_neurons,)))

        decoder_outputs_and_states = decoder(decoder_inputs, initial_state=decoder_states_inputs)

        decoder_outputs = decoder_outputs_and_states[0]
        decoder_states = decoder_outputs_and_states[1:]

        decoder_outputs = decoder_dense(decoder_outputs)

        self.decoder_predict_model = keras.models.Model([decoder_inputs] + decoder_states_inputs,
                                                   [decoder_outputs] + decoder_states)

    def train(self, data):

        # random_sine returns a generator that produces batches of training samples ([encoder_input, decoder_input], decoder_output)
        # You can play with the min max frequencies of the sine waves, the number of sine waves that are summed etc...
        # Another interesing exercise could be to see whether the model generalises well on sums of 3 signals if it's only been
        # trained on sums of 2 signals...
        # train_data_generator = random_sine(batch_size=self.batch_size,
        #                                    steps_per_epoch=self.steps_per_epoch,
        #                                    input_sequence_length=self.input_sequence_length,
        #                                    target_sequence_length=self.target_sequence_length,
        #                                    min_frequency=0.1, max_frequency=10,
        #                                    min_amplitude=0.1, max_amplitude=1,
        #                                    min_offset=-0.5, max_offset=0.5,
        #                                    num_signals=self.num_signals, seed=1969)

        train_data_generator = self.my_data_generator(data=data, batch_size=100,
                                                                     steps_per_epoch=self.steps_per_epoch,
                                                                     input_sequence_length=self.input_sequence_length,
                                                                     target_sequence_length=self.target_sequence_length)

        tensorboard = keras.callbacks.TensorBoard(log_dir=self.logs_path + "/{}".format(time()),
                                                  # histogram_freq=2,
                                                  write_graph=True,
                                                  write_images=True)

        self.model.fit_generator(train_data_generator,
                                 steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 callbacks=[tensorboard])
        self.model.save(self.saved_model_path)

        self._save()

    def predict_sequences_simple(self, data):
        # test_data_generator = random_sine(batch_size=1000,
        #                                   steps_per_epoch=self.steps_per_epoch,
        #                                   input_sequence_length=self.input_sequence_length,
        #                                   target_sequence_length=self.target_sequence_length,
        #                                   min_frequency=0.1, max_frequency=10,
        #                                   min_amplitude=0.1, max_amplitude=1,
        #                                   min_offset=-0.5, max_offset=0.5,
        #                                   num_signals=self.num_signals, seed=2000)

        test_data_generator = self.my_data_generator(data=data, batch_size=100,
                                                                     steps_per_epoch=self.steps_per_epoch,
                                                                     input_sequence_length=self.input_sequence_length,
                                                                     target_sequence_length=self.target_sequence_length)

        # (x_encoder_test2, x_decoder_test2), y_test2 = next(test_data_generator2)

        (x_encoder_test, x_decoder_test), y_test = next(test_data_generator)  # x_decoder_test is composed of zeros.

        print("(x_encoder_test, x_decoder_test), y_test", (x_encoder_test, x_decoder_test), y_test)

        y_test_predicted = self.model.predict([x_encoder_test, x_decoder_test])

        indices = np.random.choice(range(x_encoder_test.shape[0]), replace=False, size=10)

        for index in indices:
            plot_prediction(x_encoder_test[index, :, :], y_test[index, :, :], y_test_predicted[index, :, :], "lstm_test_data_" + str(index))

    def predict_sequences(self, data):
        # train_data_generator = random_sine(batch_size=1000,
        #                                    steps_per_epoch=self.steps_per_epoch,
        #                                    input_sequence_length=self.input_sequence_length,
        #                                    target_sequence_length=self.target_sequence_length,
        #                                    min_frequency=0.1, max_frequency=10,
        #                                    min_amplitude=0.1, max_amplitude=1,
        #                                    min_offset=-0.5, max_offset=0.5,
        #                                    num_signals=self.num_signals, seed=1969)

        train_data_generator = self.my_data_generator(data=data, batch_size=100,
                                                                    steps_per_epoch=self.steps_per_epoch,
                                                                    input_sequence_length=self.input_sequence_length,
                                                                    target_sequence_length=self.target_sequence_length)

        (x_train, _), y_train = next(train_data_generator)

        print("(x_train, _), y_train", x_train.shape, y_train.shape)

        y_train_predicted = self._predict(x_train, self.encoder_predict_model,
                                         self.decoder_predict_model, self.num_steps_to_predict)

        # Select 10 random examples to plot
        indices = np.random.choice(range(x_train.shape[0]), replace=False, size=10)

        for index in indices:
            plot_prediction(x_train[index, :, :], y_train[index, :, :], y_train_predicted[index, :, :], "lstm_test_data_" + str(index))

    def _predict(self, x, encoder_predict_model, decoder_predict_model, num_steps_to_predict):
        """Predict time series with encoder-decoder.

        Uses the encoder and decoder models previously trained to predict the next
        num_steps_to_predict values of the time series.

        Arguments
        ---------
        x: input time series of shape (batch_size, input_sequence_length, input_dimension).
        encoder_predict_model: The Keras encoder model.
        decoder_predict_model: The Keras decoder model.
        num_steps_to_predict: The number of steps in the future to predict

        Returns
        -------
        y_predicted: output time series for shape (batch_size, target_sequence_length,
            ouput_dimension)
        """
        y_predicted = []

        # Encode the values as a state vector
        states = encoder_predict_model.predict(x)

        # The states must be a list
        if not isinstance(states, list):
            states = [states]

        # Generate first value of the decoder input sequence
        decoder_input = np.zeros((x.shape[0], 1, 1))
        # instead feed random input, like in training.
        # decoder_input = np.random.normal(0, 1, (x.shape[0], 1, 1))

        for _ in range(num_steps_to_predict):
            outputs_and_states = decoder_predict_model.predict([decoder_input] + states, batch_size=self.batch_size)
            output = outputs_and_states[0]
            states = outputs_and_states[1:]

            # add predicted value
            y_predicted.append(output)

        return np.concatenate(y_predicted, axis=1)

    def my_data_generator(self, data, batch_size, steps_per_epoch,
                      input_sequence_length, target_sequence_length):

        while True:
            # Reset seed to obtain same sequences from epoch to epoch
            np.random.seed(42)

            for _ in range(steps_per_epoch):

                samples = self.collect_samples(data, batch_size, input_sequence_length + target_sequence_length)

                # print("samples", samples.shape)

                # samples = np.expand_dims(samples, axis=2)

                # print("samples, expand_dims", samples.shape)

                encoder_input = samples[:, :input_sequence_length, :]
                decoder_output = samples[:, input_sequence_length:, :]

                # The output of the generator must be ([encoder_input, decoder_input], [decoder_output])
                # decoder_input = np.zeros((decoder_output.shape[0], decoder_output.shape[1], 1))
                # instead of zeros we give it a random input
                decoder_input = np.random.normal(0, 1, (decoder_output.shape[0], decoder_output.shape[1], 1))

                # print("encoder_input.shape, decoder_input.shape, decoder_output.shape", encoder_input.shape, decoder_input.shape, decoder_output.shape)

                yield ([encoder_input, decoder_input], decoder_output)

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