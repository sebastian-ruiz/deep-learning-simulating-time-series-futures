from autoencoders.autoencoder_model import Autoencoder
from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import json
from helpers.evaluate import reconstruction_error
from scipy.signal import savgol_filter

def simulate():
    plotting = Plotting()
    preprocess_normalisation = PreprocessData()
    preprocess_normalisation.enable_normalisation_scaler = True
    preprocess_normalisation.feature_range = [0, 1]
    # preprocess_normalisation.enable_scaler = True


    # 1. get data and apply normalisation
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess_normalisation.get_data()

    # plotting.plot_2d(sets_training_scaled[0][:, 0], "sets_training_scaled[0][:, 0]", save=False)
    # plotting.plot_2d(sets_test_scaled[0][:, 0], "test_feature_normalised_short_end", save=True)

    all_stacked = np.vstack((np.vstack(sets_training), np.vstack(sets_test)))
    all_stacked_scaled = np.vstack((np.vstack(sets_training_scaled), np.vstack(sets_test_scaled)))
    all_training_scaled = np.vstack(sets_training_scaled)

    # print("all_stacked_scaled.shape", all_stacked_scaled.shape)
    # plotting.plot_2d(all_stacked[:, 0], "training and test data", save=False)
    # plotting.plot_2d(all_stacked_scaled[:, 0], "training and test data scaled", save=False)

    ae_params = {'input_dim': sets_training_scaled[0].shape[1], # 56
              'latent_dim': 2,
              'hidden_layers': (56, 40, 28, 12, 4, 2),
              'leaky_relu': 0.1,
              'loss': 'mse',
              'last_activation': 'linear',
              'batch_size': 20,
              'epochs': 100,
              'steps_per_epoch': 500}
    ae_params_hash = hashlib.md5(json.dumps(ae_params, sort_keys=True).encode('utf-8')).hexdigest()

    autoencoder = Autoencoder(ae_params)
    # autoencoder.train(all_stacked_scaled, sets_test_scaled)
    # autoencoder.train(sets_test_scaled[0], sets_test_scaled)
    # autoencoder.train(all_training_scaled, sets_test_scaled)
    # autoencoder.save_model("ae_" + ae_params_hash)
    autoencoder.load_model("ae_" + ae_params_hash)

    # 2: encode data using autoencoder
    sets_encoded_training = []
    for set_training_scaled in sets_training_scaled:
        sets_encoded_training.append(autoencoder.encode(set_training_scaled))

    sets_encoded_test = []
    for set_test_scaled in sets_test_scaled:
        sets_encoded_test.append(autoencoder.encode(set_test_scaled))

    plotting.plot_2d(sets_encoded_test[0], "test_feature_normalised_encoded_autoencoder_on_", save=True)

    # 6: decode using autoencoder
    decoded_test = autoencoder.decode(sets_encoded_test[0])

    # 7: undo minimax, for now only the first simulation
    simulated = preprocess_normalisation.rescale_data(decoded_test, dataset_name=test_dataset_names[0])

    plotting.plot_some_curves("test_feature_normalised_compare_autoencoder_before_rescale", sets_test_scaled[0], decoded_test,
                              [25, 50, 75, 815], maturities) # old: [25, 50, 75, 100, 600, 720, 740, 815]

    plotting.plot_some_curves("test_feature_normalised_compare_autoencoder", sets_test[0], simulated,
                              [25, 50, 75, 815], maturities) # old: [25, 50, 75, 100, 600, 720, 740, 815]

    # curve_smooth = []
    # for curve in simulated:
    #     print("curve.shape", curve.shape)
    #     curve_smooth.append(savgol_filter(curve, 23, 5))  # window size 51, polynomial order 3
    # curve_smooth = np.array(curve_smooth)

    print("reconstruction error BEFORE smoothing:")
    reconstruction_error(np.array(sets_test[0]), simulated)

    preprocess_normalisation.enable_curve_smoothing = True
    simulated = preprocess_normalisation.rescale_data(decoded_test, dataset_name=test_dataset_names[0])

    plotting.plot_some_curves("test_feature_normalised_compare_autoencoder", sets_test[0], simulated,
                              [25, 50, 75, 815], maturities) # old: [25, 50, 75, 100, 600, 720, 740, 815]

    # plotting.plot_some_curves("test_feature_normalised_compare_normalisation", sets_test[0], sets_test_scaled[0],
    #                           [25, 50, 75, 815, 100, 600, 720, 740], maturities, plot_separate=True)


    # reconstruction error
    # reconstruction_error(sets_test_scaled[0], decoded_test)
    print("reconstruction error AFTER smoothing:")
    reconstruction_error(np.array(sets_test[0]), simulated)
    # reconstruction_error(np.array(sets_test[0]), curve_smooth)



if __name__ == '__main__':
    simulate()
