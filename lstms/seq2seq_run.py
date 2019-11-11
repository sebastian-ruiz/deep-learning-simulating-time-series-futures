from lstms.seq2seq_model import Model
from helpers.preprocess_data import PreprocessData
from helpers.evaluate import *
from helpers.plotting import Plotting
import numpy as np

def main():
    preprocess1 = PreprocessData(PreprocessType.STANDARDISATION_OVER_TENORS, short_end=True)
    preprocess2 = PreprocessData(PreprocessType.LOG_RETURNS_OVER_TENORS, short_end=True)

    # 1. get data and apply scaling
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess1.get_data()

    print("sets_test_scaled, sets_training_scaled:", sets_test_scaled[0].shape, sets_training_scaled[0].shape)

    # 2: log returns of encoded data
    sets_encoded_log_training = preprocess2.scale_data(sets_training_scaled, training_dataset_names, should_fit=True)
    sets_encoded_log_test = preprocess2.scale_data(sets_test_scaled, test_dataset_names, should_fit=True)

    layers = [35, 35]  # Number of hidden neurons in each layer of the encoder and decoder

    learning_rate = 0.01
    decay = 0  # Learning rate decay

    num_input_features = 1  # The dimensionality of the input at each time step. In this case a 1D signal.
    num_output_features = 1  # The dimensionality of the output at each time step. In this case a 1D signal.
    # There is no reason for the input sequence to be of same dimension as the ouput sequence.

    loss = "mse"  # Other loss functions are possible, see Keras documentation.

    # Regularisation isn't really needed for this application
    lambda_regulariser = 0.000001  # Will not be used if regulariser is None
    regulariser = None  # Possible regulariser: keras.regularizers.l2(lambda_regulariser)

    batch_size = 512
    steps_per_epoch = 200  # batch_size * steps_per_epoch = total number of training examples
    epochs = 10

    input_sequence_length = 42  # Length of the sequence used by the encoder
    target_sequence_length = 42  # Length of the sequence predicted by the decoder
    num_steps_to_predict = 42  # Length to use when testing the model

    model = Model(layers, learning_rate, decay, num_input_features,
                  num_output_features, loss, lambda_regulariser, regulariser,
                  batch_size, steps_per_epoch, epochs, input_sequence_length,
                  target_sequence_length, num_steps_to_predict)
    model.build()
    # model.load()
    model.train(sets_encoded_log_training)
    # model.predict_sequences_simple(np.vstack(sets_training_first_last_tenors))
    model.predict_sequences(sets_encoded_log_training)


if __name__ == '__main__':
    main()