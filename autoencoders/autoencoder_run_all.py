from autoencoders.autoencoder_model import Autoencoder
from autoencoders.adversarial_autoencoder_model import AdversarialAutoencoder
from autoencoders.variational_ae_model import VariationalAutoencoder
from autoencoders.autoencoder_windows_model import AutoencoderWindows
from autoencoders.pca_model import PCAModel
from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import json
from helpers.evaluate import *
import sys
from config import Config
from timeit import default_timer as timer


def simulate(latent_dim=2, plot=False, preprocess_type=None, model_type=None, force_training=True):
    plotting = Plotting()
    preprocess = PreprocessData(preprocess_type)

    window_size = None
    if model_type is AEModel.AE_WINDOWS:
        window_size = 10


    # 1. get data and apply normalisation
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess.get_data(chunks_of=window_size)
    all_training_scaled = np.vstack(sets_training_scaled)

    if model_type is AEModel.AAE:
        ae_params = {'preprocess_type': preprocess_type.value, # only to make preprocess_type part of the hash
                     'input_dim': sets_training_scaled[0].shape[1],  # 56
                     'latent_dim': latent_dim,
                     'hidden_layers': (56, 40, 28, 12, 4,),
                     'hidden_layers_discriminator': (2, 2, ),
                     'leaky_relu': 0.1,
                     'last_activation': 'linear',
                     'last_activation_discriminator': 'sigmoid',
                     'loss_generator': 'mean_squared_error',
                     'loss_discriminator': 'binary_crossentropy',
                     'batch_size': 20,
                     'epochs': 20000}
        ae_params_hash = hashlib.md5(json.dumps(ae_params, sort_keys=True).encode('utf-8')).hexdigest()

        # 2. train/load variational autoencoder
        autoencoder = AdversarialAutoencoder(ae_params, plot=False)
    elif model_type is AEModel.VAE:
        ae_params = {'preprocess_type': preprocess_type.value, # only to make preprocess_type part of the hash
                     'input_dim': sets_training_scaled[0].shape[1],  # 56
                     'latent_dim': latent_dim,
                     'hidden_layers': (56, 40, 28, 12, 4,),
                     'leaky_relu': 0.1,
                     'last_activation': 'linear',  # sigmoid or linear
                     'loss': 'mean_squared_error',  # binary_crossentropy or mean_square_error
                     'epsilon_std': 1.0,
                     'batch_size': 20,
                     'epochs': 100,
                     'steps_per_epoch': 500}
        ae_params_hash = hashlib.md5(json.dumps(ae_params, sort_keys=True).encode('utf-8')).hexdigest()

        # 2. train/load variational autoencoder
        autoencoder = VariationalAutoencoder(ae_params, plot=False)
    elif model_type is AEModel.AE:
        ae_params = {'preprocess_type': preprocess_type.value, # only to make preprocess_type part of the hash
                     'input_dim': sets_training_scaled[0].shape[1], # 56
                     'latent_dim': latent_dim,
                     'hidden_layers': (56, 40, 28, 12, 4,),
                     'leaky_relu': 0.1,
                     'loss': 'mse',
                     'last_activation': 'linear',
                     'batch_size': 20,
                     'epochs': 100,
                     'steps_per_epoch': 500}
        ae_params_hash = hashlib.md5(json.dumps(ae_params, sort_keys=True).encode('utf-8')).hexdigest()
        autoencoder = Autoencoder(ae_params, plot=False)
    elif model_type is AEModel.PCA:
        ae_params = {'preprocess_type': preprocess_type.value,  # only to make preprocess_type part of the hash
                     'latent_dim': latent_dim}
        ae_params_hash = hashlib.md5(json.dumps(ae_params, sort_keys=True).encode('utf-8')).hexdigest()
        autoencoder = PCAModel(ae_params, plot=False)
    else: # model_type is AEModel.AE_WINDOWS:
        ae_params = {'input_dim': (window_size, sets_training_scaled[0].shape[1],),  # 10 x 56
                     'latent_dim': (2, 56,),
                     'hidden_layers': (12 * 56, 4 * 56,),
                     'leaky_relu': 0.1,
                     'loss': 'mse',
                     'last_activation': 'linear',
                     'batch_size': 20,
                     'epochs': 10,
                     'steps_per_epoch': 500, }
        ae_params_hash = hashlib.md5(json.dumps(ae_params, sort_keys=True).encode('utf-8')).hexdigest()
        autoencoder = AutoencoderWindows(ae_params, plot=False)

    if force_training:
        autoencoder.train(all_training_scaled, sets_test_scaled, "ae_" + ae_params_hash)
    else:
        autoencoder.load_else_train(all_training_scaled, sets_test_scaled, "ae_" + ae_params_hash)

    # 2: encode data using autoencoder
    sets_encoded_training = autoencoder.encode(sets_training_scaled)
    sets_encoded_test = autoencoder.encode(sets_test_scaled)

    # 6: decode using autoencoder
    decoded_test = autoencoder.decode(sets_encoded_test[0])

    # 7: undo scaling
    # decoded_generated_segments_first_sim = decoded_generated_segments[0]
    simulated = preprocess.rescale_data(decoded_test, dataset_name=test_dataset_names[0])

    preprocess.enable_curve_smoothing = True
    simulated_smooth = preprocess.rescale_data(decoded_test, dataset_name=test_dataset_names[0])

    # reconstruction error
    # error = reconstruction_error(np.array(sets_test[0]), simulated)
    # error_smooth = reconstruction_error(np.array(sets_test[0]), simulated_smooth)

    smape_result = smape(simulated, np.array(sets_test[0]), over_curves=True)
    smape_result_smooth = smape(simulated_smooth, np.array(sets_test[0]), over_curves=True)

    print(np.mean(smape_result_smooth))

    if plot and model_type is not AEModel.AE_WINDOWS:

        plotting.plot_2d(sets_encoded_test[0], preprocess_type.name + "_" + model_type.name + "_latent_space", sets_test_scaled[0].index.values, save=True)

        plotting.plot_some_curves(preprocess_type.name + "_" + model_type.name + "_in_vs_out", sets_test[0], simulated,
                                  [25, 50, 75, 815], maturities)

        # plotting.plot_some_curves("normalised_compare_ae", sets_test[0], sets_test_scaled[0],
        #                           [25, 50, 75, 815, 100, 600, 720, 740], maturities, plot_separate=True)

        preprocess.enable_curve_smoothing = False
        if model_type is AEModel.VAE:
            plotting.plot_grid_2dim(maturities, autoencoder.generator_model, preprocess_type.name + "_" + model_type.name + "_latent_grid", preprocess, test_dataset_names[0], n=6)
        elif model_type is AEModel.AAE:
            plotting.plot_grid_2dim(maturities, autoencoder.decoder, preprocess_type.name + "_" + model_type.name + "_latent_grid", preprocess, test_dataset_names[0], n=6)

    return smape_result_smooth


def simulate_and_log(preprocess_type, model_type, hidden_dim, force_training=True, plot=False, trials=5):
    np.set_printoptions(threshold=np.inf)
    config = Config()
    error = []

    filename = 'out_' + preprocess_type.name + '_' + model_type.name + '_' + str(hidden_dim) + '.txt'
    if not config.file_exists(config.get_filepath_autoencoder_logs(filename)):
        orig_stdout = sys.stdout
        f = open(config.get_filepath_autoencoder_logs(filename), 'w')
        sys.stdout = Tee(sys.stdout, f)
        for i in np.arange(trials):
            start = timer()
            error.append(simulate(latent_dim=hidden_dim, plot=plot, preprocess_type=preprocess_type,
                                  model_type=model_type, force_training=force_training))
            end = timer()
            print("Trial:" + str(i) + "time elapsed: " + str(end - start))  # Time in seconds

        sys.stdout = orig_stdout
        f.close()

        orig_stdout = sys.stdout
        f = open(config.get_filepath_autoencoder_logs('results.txt'), 'a')
        sys.stdout = Tee(sys.stdout, f)
        print("TEST:", hidden_dim, preprocess_type.name, model_type.name)
        error = np.array(error)
        error_without_nans = error[~np.isnan(error).any(axis=1)]
        print("mean:", np.mean(error_without_nans))
        print("std:", np.mean(np.std(error_without_nans, axis=1), axis=0))
        print("")
        sys.stdout = orig_stdout
        f.close()


def simulate_many(force_training=True):
    for preprocess_type in [PreprocessType.NORMALISATION_OVER_TENORS,
                            PreprocessType.STANDARDISATION_OVER_TENORS,
                            PreprocessType.LOG_RETURNS_OVER_TENORS]:
        for model_type in [AEModel.PCA, AEModel.AE, AEModel.AAE, AEModel.VAE, AEModel.AE_WINDOWS]:
            for hidden_dim in [1, 2, 3, 4]:
                simulate_and_log(preprocess_type, model_type, hidden_dim, force_training)


def simulate_one():
    error = simulate(latent_dim=2,
                     plot=True,
                     preprocess_type=PreprocessType.NORMALISATION_OVER_TENORS,
                     model_type=AEModel.AE,
                     force_training=False)
    print("error", error)


def simulate_some():
    for model_type in [AEModel.AE, AEModel.AAE, AEModel.VAE, AEModel.PCA, AEModel.AE_WINDOWS]:
        simulate_and_log(PreprocessType.LOG_RETURNS_OVER_TENORS,
                         model_type,
                         hidden_dim=2,
                         force_training=False,
                         plot=True,
                         trials=1)

        print("\n ===================================================================\n")


if __name__ == '__main__':
    # simulate_one()
    simulate_some()
    # simulate_many(force_training=True) # `force_training=False` for testing only