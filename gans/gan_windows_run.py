from gans.gan_3d_model import GAN
from autoencoders.autoencoder_windows_model import Autoencoder
from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
import numpy as np
import hashlib
import json
import matplotlib.pyplot as plt
import random
from helpers.evaluate import *
from scipy.signal import savgol_filter

def simulate():
    plotting = Plotting()
    preprocess_normalisation = PreprocessData()
    preprocess_logreturns = PreprocessData()
    preprocess_normalisation.enable_normalisation_scaler = True
    preprocess_logreturns.enable_log_returns = True

    # 1. get data and apply pre-processing
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess_normalisation.get_data()

    ae_params = { 'preprocess_type': PreprocessType.NORMALISATION_OVER_TENORS.value,
                  'input_dim': (10, sets_training_scaled[0].shape[1],), # 56
                  'latent_dim': 2*56,
                  'hidden_layers': (12*56, 4*56, ),
                  'leaky_relu': 0.1,
                  'loss': 'mse',
                  'last_activation': 'linear',
                  'batch_size': 5,
                  'epochs': 5,
                  'steps_per_epoch': 500}

    ae_params_hash = hashlib.md5(json.dumps(ae_params, sort_keys=True).encode('utf-8')).hexdigest()

    autoencoder = Autoencoder(ae_params)
    # autoencoder.train(np.vstack(sets_training_scaled), sets_test_scaled)
    # autoencoder.save_model("ae_" + ae_params_hash)
    autoencoder.load_else_train(sets_training_scaled, sets_test_scaled, "ae_" + ae_params_hash)

    # 2: encode data using autoencoder
    sets_encoded_training = autoencoder.encode(sets_training_scaled)
    sets_encoded_test = autoencoder.encode(sets_test_scaled)

    print("sets_encoded_test", sets_encoded_test[0].shape)
    plotting.plot_2d(sets_encoded_test[0], "encoded test data with deep autoencoder", save=False)

    # 3: log returns of encoded data
    sets_encoded_log_training = preprocess_logreturns.scale_data(sets_encoded_training)
    sets_encoded_log_test = preprocess_logreturns.scale_data(sets_encoded_test)

    plotting.plot_2d(sets_encoded_log_test[0], "encoded test data with deep autoencoder, then log returns", save=False)

    num_c = 6*7
    num_o = 6*7
    gan_params = {'ae_params_hash': ae_params_hash,
                  'num_tenors': sets_encoded_log_training[0].shape[1],
                  'num_c': num_c,
                  'num_z': 6*7,
                  'num_o': num_o,
                  'gen_model_type': 'standard', # conv
                  'dis_model_type': 'standard', # conv
                  'gen_layers': (4*(6*7*2),), # 4 * num_o * num_tenors
                  'dis_layers': (4*(6*7),), # 4 * num_o
                  'gen_last_activation': 'tanh',
                  'dis_last_activation': 'sigmoid',
                  'loss': 'binary_crossentropy',
                  'batch_size': 128,
                  'epochs': 20000}
    gan_params_hash = hashlib.md5(json.dumps(gan_params, sort_keys=True).encode('utf-8')).hexdigest()

    gan = GAN(gan_params)  # try training on larger input and output
    # gan.train(sets_encoded_log_training, sample_interval=200)
    # gan.save_model("gan_" + gan_params_hash)
    gan.load_model("gan_" + gan_params_hash)

    # COV TEST, TEMPORARY
    # for name, set in zip(training_dataset_names, sets_training):
    #     print("name:", name)
    #     set_cov_log_returns_over_features = cov_log_returns_over_features(set)
    #     plotting.plot_3d_cov("covariance_time_series_" + name, set_cov_log_returns_over_features, show_title=False)
    #     plotting.plot_3d("time_series_" + name, set, maturities)
    # END COV TEST.

    # 4: simulate on encoded log returns, conditioned on test dataset
    num_simulations = 10
    num_repeats = 0
    generated, _ = gan.generate(condition=sets_encoded_log_test[-1], condition_on_end=False, num_simulations=num_simulations, repeat=num_repeats)

    # insert the last real futures curve in order to do rescaling
    print("sets_encoded_log_test[-1][num_c] shape", sets_encoded_log_test[-1].iloc[num_c].shape)
    print("generated_segments.shape", generated.shape)
    generated = np.insert(generated, 0, sets_encoded_log_test[-1].iloc[num_c], axis=0)

    # 5: undo log-returns # todo: this start_value is actually one off! Error still persists... autoencoder causing the difference?
    encoded_generated = preprocess_logreturns.rescale_data(generated, start_value=sets_encoded_test[-1][num_c])
    encoded_generated = encoded_generated[:, 1:] # remove first curve again
    # 6: decode using autoencoder
    decoded_generated_segments = autoencoder.decode(encoded_generated)

    # 7: undo minimax, for now only the first simulation
    simulated = preprocess_normalisation.rescale_data(decoded_generated_segments, dataset_name=test_dataset_names[-1])

    preprocess_normalisation.enable_curve_smoothing = True
    simulated_smooth = preprocess_normalisation.rescale_data(decoded_generated_segments, dataset_name=test_dataset_names[-1])

    real = np.array(sets_test[-1])[num_c:num_c + num_o]

    print("simulated, real", simulated.shape, real.shape)

    smape_result = smape(simulated, real)
    smape_result_smooth = smape(simulated_smooth, real)
    print("smape_result and smooth", smape_result, smape_result_smooth)
    print("smape_resul_smooth", smape_result_smooth)

    #     # plotting.plot_3d_training("3d recursively generated with GAN", generated_segments, real_segment, show=True)
    #     # plotting.plot_2d(sim, "ae_gan/ae_gan" + rand_name + "_sim", timeseries=True, save=True, title=True)
    #     # plotting.plot_2d(real, "ae_gan/ae_gan" + rand_name + "_real", timeseries=True, save=True, title=True)
    #
    #     # # ignore if a simulation produced negative values
    #     # if (simulated_time_series > 0).all():
    #     #
    #     #     smoothed_simulated_time_series = savgol_filter(simulated_time_series, 23, 5)
    #     #
    #     #     print("smoothed_simulated_time_series.shape", smoothed_simulated_time_series.shape)
    #     #
    #     #     plotting.plot_3d_training("ae_gan/ae_gan" + rand_name, smoothed_simulated_time_series, sets_test[-1], show=True, after_real_data=True)
    #     #     # plotting.plotly_3d("ae_gan/ae_gan", simulated, sets_test[-1])
    #     #
    #     #     set_cov_log_returns_over_features = cov_log_returns_over_features(smoothed_simulated_time_series)
    #     #     plotting.plot_3d_cov("ae_gan/ae_gan" + rand_name + "_cov", set_cov_log_returns_over_features, show_title=False)
    #
    # smape_results = np.array(smape_results)
    # print("smape mean and variance:", np.mean(smape_results), np.var(smape_results))




if __name__ == '__main__':
    simulate()
