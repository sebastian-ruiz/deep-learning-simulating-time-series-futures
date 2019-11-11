from autoencoders.autoencoder_windows_model import AutoencoderWindows
from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import json
from helpers.evaluate import *
import os, sys

def simulate(plot=True):
    plotting = Plotting()
    preprocess = PreprocessData()
    preprocess.enable_normalisation_scaler = True
    preprocess.feature_range = [0, 1]

    window_size = 20

    # 1. get data and apply normalisation
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess.get_data(chunks_of=window_size)

    print("sets_training_scaled.shape", sets_training_scaled[0].shape)

    # plotting.plot_2d(sets_training_scaled[0][:, 0], "sets_training_scaled[0][:, 0]", save=False)
    # plotting.plot_2d(sets_test_scaled[0][:, 0], "test_feature_normalised_short_end", save=True)

    ae_params = {'input_dim': (window_size, sets_training_scaled[0].shape[1],), # 10 x 56
              'latent_dim': (2,56,),
              'hidden_layers': (12*56, 4*56, ),
              'leaky_relu': 0.1,
              'loss': 'mse',
              'last_activation': 'linear',
              'batch_size': 20,
              'epochs': 100,
              'steps_per_epoch': 500, }
    ae_params_hash = hashlib.md5(json.dumps(ae_params, sort_keys=True).encode('utf-8')).hexdigest()

    autoencoder = AutoencoderWindows(ae_params)

    print("sets_training_scaled", sets_training_scaled[0].shape)

    autoencoder.train(sets_training_scaled, sets_test_scaled)
    autoencoder.save_model("ae_" + ae_params_hash)
    # autoencoder.load_model("ae_" + ae_params_hash)

    # 2: encode data using autoencoder
    sets_encoded_training = []
    for set_training_scaled in sets_training_scaled:
        sets_encoded_training.append(autoencoder.encode(set_training_scaled))

    sets_encoded_test = []
    for set_test_scaled in sets_test_scaled:
        sets_encoded_test.append(autoencoder.encode(set_test_scaled))

    print("sets_encoded_training", len(sets_encoded_training), sets_encoded_training[0].shape)
    print("sets_encoded_test", sets_encoded_test[0].shape)

    # 6: decode using autoencoder
    decoded_test = autoencoder.decode(sets_encoded_test[0])

    print("decoded_test", decoded_test.shape)

    # 7: undo minimax, for now only the first simulation
    # decoded_generated_segments_first_sim = decoded_generated_segments[0]
    preprocess.enable_curve_smoothing = True
    simulated_smooth = preprocess.rescale_data(decoded_test, dataset_name=test_dataset_names[0])

    # reconstruction error
    # reconstruction_error(sets_test_scaled[0], decoded_test)
    # error = reconstruction_error(np.array(sets_test[0]), simulated_smooth)
    # print("error:", error)

    smape_result_smooth = smape(simulated_smooth, np.array(sets_test[0]), over_curves=True)

    print(np.mean(smape_result_smooth), np.var(smape_result_smooth))

    if plot:
        # plotting.plot_2d(sets_encoded_test[0], "test_feature_normalised_encoded_autoencoder_on_", save=True)

        # plotting.plot_some_curves("normalised_compare_ae_before_rescale", sets_test_scaled[0], decoded_test,
        #                           [25, 50, 75, 815], maturities)

        plotting.plot_some_curves("normalised_compare_ae", sets_test[0], simulated_smooth,
                                  [25, 50, 75, 815], maturities)

        # plotting.plot_some_curves("normalised_compare_ae", sets_test[0], sets_test_scaled[0],
        #                           [25, 50, 75, 815, 100, 600, 720, 740], maturities, plot_separate=True)

    # return error


if __name__ == '__main__':
    simulate()
