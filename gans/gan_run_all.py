from gans.gan_3d_model import GAN
from gans.cwgan_gp_model import CWGANGP
from autoencoders.autoencoder_model import Autoencoder
from autoencoders.adversarial_autoencoder_model import AdversarialAutoencoder
from autoencoders.variational_ae_model import VariationalAutoencoder
from autoencoders.pca_model import PCAModel
from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
import numpy as np
import hashlib
import json
import matplotlib.pyplot as plt
import random
from helpers.evaluate import *
from scipy.signal import savgol_filter
import sys
from config import Config
from timeit import default_timer as timer


def simulate(latent_dim=2, preprocess_type1=None, preprocess_type2=None, ae_model=None, gan_model=None, force_training=True, plot=False):
    preprocess1 = PreprocessData(preprocess_type1)
    preprocess2 = PreprocessData(preprocess_type2)

    # 1. get data and apply scaling
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess1.get_data()

    if ae_model is AEModel.AAE:
        ae_params = {'preprocess_type': preprocess_type1.value, # only to make preprocess_type part of the hash
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
        autoencoder = AdversarialAutoencoder(ae_params, plot=False)
    elif ae_model is AEModel.VAE:
        ae_params = {'preprocess_type': preprocess_type1.value, # only to make preprocess_type part of the hash
                     'input_dim': sets_training_scaled[0].shape[1],  # 56
                     'latent_dim': latent_dim,
                     'hidden_layers': (56, 40, 28, 12, 4,),
                     'leaky_relu': 0.1,
                     'last_activation': 'linear',  # sigmoid or linear
                     'loss': 'mean_square_error',  # binary_crossentropy or mean_square_error
                     'epsilon_std': 1.0,
                     'batch_size': 20,
                     'epochs': 100,
                     'steps_per_epoch': 500}
        ae_params_hash = hashlib.md5(json.dumps(ae_params, sort_keys=True).encode('utf-8')).hexdigest()
        autoencoder = VariationalAutoencoder(ae_params, plot=False)
    elif ae_model is AEModel.AE:
        ae_params = {'preprocess_type': preprocess_type1.value, # only to make preprocess_type part of the hash
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
    else: # elif ae_model is AEModel.PCA:
        ae_params = {'preprocess_type': preprocess_type1.value,  # only to make preprocess_type part of the hash
                     'latent_dim': latent_dim }
        ae_params_hash = hashlib.md5(json.dumps(ae_params, sort_keys=True).encode('utf-8')).hexdigest()
        autoencoder = PCAModel(ae_params, plot=False)

    # 2. train/load autoencoder
    autoencoder.load_else_train(np.vstack(sets_training_scaled), sets_test_scaled, "ae_" + ae_params_hash)

    # 2: encode data using autoencoder
    sets_encoded_training = autoencoder.encode(sets_training_scaled)
    sets_encoded_test = autoencoder.encode(sets_test_scaled)

    # 3: log returns of encoded data
    sets_encoded_log_training = preprocess2.scale_data(sets_encoded_training, training_dataset_names, should_fit=True)
    sets_encoded_log_test = preprocess2.scale_data(sets_encoded_test, test_dataset_names, should_fit=True)

    num_z = 6 * 7
    num_c = 6 * 7
    num_o = 6 * 7
    if gan_model is GANModel.WGAN:
        gan_params = {'ae_params_hash': ae_params_hash,
                      'num_tenors': sets_encoded_log_training[0].shape[1],
                      'num_c': num_c,
                      'num_z': num_z,
                      'num_o': num_o,
                      'gen_model_type': 'standard',  # conv
                      'dis_model_type': 'standard',  # conv
                      'gen_layers': (4 * (6 * 7 * 2),),  # 4 * num_o * num_tenors
                      'dis_layers': (4 * (6 * 7),),  # 4 * num_o
                      'gen_last_activation': 'tanh',
                      'dis_last_activation': 'sigmoid',
                      'loss': 'binary_crossentropy',
                      'batch_size': 32,
                      'epochs': 10000,
                      'sample_interval': 1000}
        gan_params_hash = hashlib.md5(json.dumps(gan_params, sort_keys=True).encode('utf-8')).hexdigest()
        gan = CWGANGP(gan_params, plot=False)
    else:
        if gan_model is GANModel.GAN_CONV:
            model_type = 'conv'
        else: # if gan_model is GANModel.GAN:
            model_type = 'standard'

        gan_params = {'ae_params_hash': ae_params_hash,
                  'num_tenors': sets_encoded_log_training[0].shape[1],
                  'num_c': num_c,
                  'num_z': num_z,
                  'num_o': num_o,
                  'gen_model_type': model_type, # conv
                  'dis_model_type': model_type, # conv
                  'gen_layers': (4*(6*7*2),), # 4 * num_o * num_tenors
                  'dis_layers': (4*(6*7),), # 4 * num_o
                  'gen_last_activation': 'tanh',
                  'dis_last_activation': 'sigmoid',
                  'loss': 'binary_crossentropy',
                  'batch_size': 128,
                  'epochs': 20000}
        gan_params_hash = hashlib.md5(json.dumps(gan_params, sort_keys=True).encode('utf-8')).hexdigest()
        gan = GAN(gan_params, plot=False)  # try training on larger input and output

    if force_training:
        gan.train(sets_encoded_log_training, "gan_" + gan_params_hash)
    else:
        gan.load_else_train(sets_encoded_log_training, "gan_" + gan_params_hash)

    # 4: simulate on encoded log returns, conditioned on test dataset
    num_simulations = 100
    num_repeats = 1
    generated, _ = gan.generate(condition=sets_encoded_log_test[-1], condition_on_end=False,
                                num_simulations=num_simulations, repeat=num_repeats)

    # insert the last real futures curve in order to do rescaling
    if preprocess_type2 is PreprocessType.LOG_RETURNS_OVER_TENORS:
        generated = np.insert(generated, 0, sets_encoded_log_test[-1].iloc[num_c], axis=1)

    # 5: undo scaling
    encoded_generated = preprocess2.rescale_data(generated, start_value=sets_encoded_test[-1][num_c], dataset_name=test_dataset_names[-1])
    if preprocess_type2 is PreprocessType.LOG_RETURNS_OVER_TENORS:
        encoded_generated = encoded_generated[:, 1:]  # remove first curve again

    # 6: decode using autoencoder
    decoded_generated_segments = autoencoder.decode(encoded_generated)

    # 7: undo scaling, this can be log-returns
    simulated = preprocess1.rescale_data(decoded_generated_segments, start_value=sets_test[-1].iloc[num_c], dataset_name=test_dataset_names[-1])

    preprocess1.enable_curve_smoothing = True
    simulated_smooth = preprocess1.rescale_data(decoded_generated_segments, start_value=sets_test[-1].iloc[num_c], dataset_name=test_dataset_names[-1])

    if preprocess_type2 is PreprocessType.LOG_RETURNS_OVER_TENORS:
        real = sets_test[-1].iloc[num_c:num_c + num_o * num_repeats + 1] # `+1` because the log-returns also does +1
    else:
        real = sets_test[-1].iloc[num_c:num_c + num_o * num_repeats + 1]

    print("simulated, real", simulated.shape, real.shape)

    smape_result = smape(simulated, real)
    smape_result_smooth = smape(simulated_smooth, real)

    print("smape_result_smooth mean and std:", np.mean(smape_result_smooth), np.std(smape_result_smooth))

    if plot:
        plotting = Plotting()
        plotting.plot_3d("real", real, show_title=False)

        cov_log_returns = cov_log_returns_over_tenors(real)
        plotting.plot_3d_cov("gan_real_cov", cov_log_returns, show_title=False)

        for i in np.arange(1, 11):
            # name =  '_' + preprocess_type1.name + '_' + preprocess_type2.name + '_' + str(latent_dim) + '_' + ae_model.name + '_'+ gan_model.name
            plotting.plot_3d("gan_simulated_" + str(i), simulated_smooth[i], maturities=maturities, time=real.index.values, show_title=False)
            smape_result = smape(simulated_smooth[i], real)
            print("simulated_smooth[i], real", simulated_smooth[i].shape, real.shape)
            print("simulate rates", i)
            print("smape:", smape_result)
            print("=============\n")

            cov_log_returns = cov_log_returns_over_tenors(simulated_smooth[i])
            plotting.plot_3d_cov("gan_simulated_" + str(i) + "_cov", cov_log_returns, maturities=maturities, show_title=False)

    return smape_result_smooth


def simulate_and_log(preprocess_type1, preprocess_type2, hidden_dim, ae_model, gan_model, force_training=True, plot=False):
    config = Config()
    error = []

    filename = preprocess_type1.name + '_' + preprocess_type2.name + '_' + str(hidden_dim) + '_' + ae_model.name + '_'+ gan_model.name + '.txt'
    if not config.file_exists(config.get_filepath_gan_logs(filename)):
        orig_stdout = sys.stdout
        f = open(config.get_filepath_gan_logs(filename), 'w')
        sys.stdout = Tee(sys.stdout, f)
        print("TEST:", filename)
        for i in np.arange(5):
            start = timer()
            error.append(
                simulate(latent_dim=hidden_dim, preprocess_type1=preprocess_type1, preprocess_type2=preprocess_type2,
                         ae_model=ae_model, gan_model=gan_model, force_training=force_training, plot=plot))
            end = timer()
            print("Trial:" + str(i) + "time elapsed: " + str(end - start))  # Time in seconds

        sys.stdout = orig_stdout
        f.close()

        orig_stdout = sys.stdout
        f = open(config.get_filepath_gan_logs('results.txt'), 'a')
        sys.stdout = Tee(sys.stdout, f)
        print("TEST:", hidden_dim, preprocess_type1.name, preprocess_type2.name, ae_model.name, gan_model.name)
        error = np.array(error)
        error_without_nans = error[~np.isnan(error).any(axis=1)]
        print("mean:", np.mean(error_without_nans))
        print("std:", np.mean(np.std(error_without_nans, axis=1), axis=0))
        print("")
        sys.stdout = orig_stdout
        f.close()


def simulate_many(force_training=True):

    for preprocess_type in [PreprocessType.STANDARDISATION_OVER_TENORS]:
        for ae_model in [AEModel.PCA, AEModel.AE, AEModel.AAE]:
            for gan_model in [GANModel.GAN, GANModel.GAN_CONV, GANModel.WGAN]:
                for hidden_dim in [2]:
                    simulate_and_log(preprocess_type, PreprocessType.LOG_RETURNS_OVER_TENORS, hidden_dim, ae_model, gan_model, force_training)


    # for preprocess_type in [PreprocessType.NORMALISATION_OVER_TENORS, PreprocessType.STANDARDISATION_OVER_TENORS]:
    #     for ae_model in [AEModel.AE, AEModel.AAE, AEModel.VAE, AEModel.PCA]:
    #         for gan_model in [GANModel.GAN, GANModel.GAN_CONV, GANModel.WGAN]:
    #             for hidden_dim in [1, 2, 3, 4]:
    #                 simulate_and_log(preprocess_type, PreprocessType.LOG_RETURNS_OVER_TENORS, hidden_dim, ae_model, gan_model, force_training)

    # for preprocess_type in [PreprocessType.NORMALISATION_OVER_TENORS,
    #                         PreprocessType.STANDARDISATION_OVER_TENORS,
    #                         PreprocessType.NONE]:
    #     for ae_model in [AEModel.AE, AEModel.AAE, AEModel.VAE, AEModel.PCA]:
    #         for gan_model in [GANModel.GAN, GANModel.GAN_CONV, GANModel.WGAN]:
    #             for hidden_dim in [1, 2, 3, 4]:
    #                 simulate_and_log(PreprocessType.LOG_RETURNS_OVER_TENORS, preprocess_type, hidden_dim, ae_model, gan_model, force_training)


def simulate_one():

    simulate(latent_dim=2,
             preprocess_type1=PreprocessType.STANDARDISATION_OVER_TENORS,
             preprocess_type2=PreprocessType.LOG_RETURNS_OVER_TENORS,
             ae_model=AEModel.PCA,
             gan_model=GANModel.GAN,
             force_training=False,
             plot=True)

    # simulate(latent_dim=2,
    #          preprocess_type1=PreprocessType.STANDARDISATION_OVER_TENORS,
    #          preprocess_type2=PreprocessType.LOG_RETURNS_OVER_TENORS,
    #          ae_model=AEModel.AE,
    #          gan_model=GANModel.GAN_CONV,
    #          force_training=False,
    #          plot=True)



if __name__ == '__main__':
    simulate_one()
    # simulate_many(force_training=True)  # `force_training=False` for testing only
