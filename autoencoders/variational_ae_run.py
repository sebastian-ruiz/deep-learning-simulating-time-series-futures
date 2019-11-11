from autoencoders.variational_ae_model import VariationalAutoencoder
from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
from config import Config
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import hashlib
import json
from helpers.evaluate import reconstruction_error

def simulate():
    plotting = Plotting()
    preprocess_normalisation = PreprocessData()
    preprocess_normalisation.enable_normalisation_scaler = True
    preprocess_normalisation.feature_range = [-1, 1]
    # preprocess_normalisation.enable_ignore_price = True

    # 1. get data and apply normalisation
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess_normalisation.get_data()
    all_training_scaled = np.vstack(sets_training_scaled)

    ae_params = {'input_dim': sets_training_scaled[0].shape[1], # 56
              'latent_dim': 3,
              'hidden_layers': (56, 40, 28, 12, 4, ),
              'leaky_relu': 0.1,
              'last_activation': 'linear', # sigmoid or linear
              'loss': 'mean_square_error', # binary_crossentropy or mean_square_error
              'epsilon_std' : 1.0,
              'batch_size': 20,
              'epochs': 100,
              'steps_per_epoch': 500}
    ae_params_hash = hashlib.md5(json.dumps(ae_params, sort_keys=True).encode('utf-8')).hexdigest()

    # 2. train/load variational autoencoder
    vae = VariationalAutoencoder(ae_params)

    vae.train(all_training_scaled, sets_test_scaled)
    vae.save_model("vae_" + ae_params_hash)
    # vae.load_model("vae_" + ae_params_hash)

    # 3: encode data using autoencoder
    sets_encoded_training = []
    for set_training_scaled in sets_training_scaled:
        sets_encoded_training.append(vae.encode(set_training_scaled))

    sets_encoded_test = []
    for set_test_scaled in sets_test_scaled:
        sets_encoded_test.append(vae.encode(set_test_scaled))

    # 4: decode using vae
    decoded_data = vae.decode(sets_encoded_test[0])

    # 7: undo minimax, for now only the first simulation
    simulated = preprocess_normalisation.rescale_data(decoded_data, dataset_name=test_dataset_names[0])

    # reconstruction error
    # reconstruction_error(sets_test_scaled[0], decoded_data)
    reconstruction_error(np.array(sets_test[0]), simulated)

    # plot latent space
    plotting.plot_2d(sets_encoded_test[0], "test_feature_normalised_encoded_vae_on_", save=True)
    plotting.plot_space(maturities, vae, "variational_grid", latent_dim=sets_encoded_test[0].shape[1])

    # plot scaled results
    plotting.plot_some_curves("test_feature_normalised_compare_vae_scaled", sets_test_scaled[0], decoded_data,
                              [25, 50, 75, 815], maturities)

    plotting.plot_some_curves("test_feature_normalised_compare_vae", sets_test[0], simulated,
                              [25, 50, 75, 815], maturities)

if __name__ == '__main__':
    simulate()
