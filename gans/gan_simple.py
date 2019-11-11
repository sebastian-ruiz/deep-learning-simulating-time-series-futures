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
    preprocess1 = PreprocessData(preprocess_type1, short_end=True)
    preprocess2 = PreprocessData(preprocess_type2, short_end=True)

    # 1. get data and apply scaling
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess1.get_data()

    print("sets_test_scaled, sets_training_scaled:", sets_test_scaled[0].shape, sets_training_scaled[0].shape)

    # 2: log returns of encoded data
    sets_encoded_log_training = preprocess2.scale_data(sets_training_scaled, training_dataset_names, should_fit=True)
    sets_encoded_log_test = preprocess2.scale_data(sets_test_scaled, test_dataset_names, should_fit=True)

    num_c = 6*7
    num_o = 6*7
    if gan_model is GANModel.WGAN:
        gan_params = {'short_end_encoding': preprocess_type1.name + "_" + preprocess_type2.name,
                      'num_tenors': sets_encoded_log_training[0].shape[1],
                      'num_c': 6 * 7,
                      'num_z': 6 * 7,
                      'num_o': 6 * 7,
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

        print("num tenors:", sets_encoded_log_training[0].shape[1])

        gan_params = {'short_end_encoding': preprocess_type1.name + "_" + preprocess_type2.name,
                  'num_tenors': sets_encoded_log_training[0].shape[1],
                  'num_c': num_c,
                  'num_z': 6*7,
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
    num_repeats = 0

    print("sets_encoded_log_test[-1]", sets_encoded_log_test[-1].shape)

    generated, _ = gan.generate(condition=sets_encoded_log_test[-1], condition_on_end=False,
                                num_simulations=num_simulations, repeat=num_repeats)

    # insert the last real futures curve in order to do rescaling
    if preprocess_type2 is PreprocessType.LOG_RETURNS_OVER_TENORS:
        generated = np.insert(generated, 0, sets_encoded_log_test[-1].iloc[num_c], axis=1)

    print("sets_test_scaled[-1]", sets_test_scaled[-1].shape)
    print("sets_test_scaled[-1][num_c]", sets_test_scaled[-1].iloc[num_c])

    # 5: undo scaling
    encoded_generated = preprocess2.rescale_data(generated, start_value=sets_test_scaled[-1].iloc[num_c], dataset_name=test_dataset_names[-1])
    if preprocess_type2 is PreprocessType.LOG_RETURNS_OVER_TENORS:
        encoded_generated = encoded_generated[:, 1:]  # remove first curve again

    # 7: undo scaling, this can be log-returns
    simulated = preprocess1.rescale_data(encoded_generated, start_value=sets_test[-1].iloc[num_c], dataset_name=test_dataset_names[-1])

    if preprocess_type2 is PreprocessType.LOG_RETURNS_OVER_TENORS:
        real = np.array(sets_test[-1])[num_c:num_c + num_o + 1] # `+1` because the log-returns also does +1
    else:
        real = np.array(sets_test[-1])[num_c:num_c + num_o + 1]

    sim = simulated.reshape(100, 43)

    print("sets_test[-1].iloc[num_c], sim[0][0]", sets_test[-1].iloc[num_c], sim[0][0], sim[1][0], sim[2][0])
    print("real, simulated", real.shape, sim.shape)

    smape_result = smape(sim, real, over_curves=True)

    if plot:
        condition_and_real = sets_test[-1].iloc[0:num_c + num_o + 1]
        plotting = Plotting()
        plotting.plot_training_sample("simulated_simple", sim, condition_and_real, num_c, after_real_data=True)


        # print("smape test:", smape(simulated[0], real), smape_result)


    return smape_result


def simulate_and_log(preprocess_type1, preprocess_type2, hidden_dim, ae_model, gan_model, force_training=True, plot=False):
    config = Config()
    error = []

    filename = 'gan_simple_' + preprocess_type1.name + '_' + preprocess_type2.name + '_' + str(hidden_dim) + '_' + ae_model.name + '_'+ gan_model.name + '.txt'
    if not config.file_exists(config.get_filepath_gan_logs(filename)):
        orig_stdout = sys.stdout
        f = open(config.get_filepath_gan_logs('simple/' + filename), 'w')
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
        f = open(config.get_filepath_gan_logs('simple/results.txt'), 'a')
        sys.stdout = Tee(sys.stdout, f)
        print("TEST:", hidden_dim, preprocess_type1.name, preprocess_type2.name, ae_model.name, gan_model.name)
        error = np.array(error)
        error_without_nans = error[~np.isnan(error).any(axis=1)]
        print("mean:", np.mean(error_without_nans))
        print("var:", np.mean(np.var(error_without_nans, axis=1), axis=0))
        print("")
        sys.stdout = orig_stdout
        f.close()


def simulate_many(force_training=True):

    for gan_model in [GANModel.GAN, GANModel.WGAN]:
        for hidden_dim in [2]:
            simulate_and_log(PreprocessType.STANDARDISATION_OVER_TENORS,
                             PreprocessType.LOG_RETURNS_OVER_TENORS,
                             hidden_dim,
                             AEModel.AE,
                             gan_model,
                             force_training)


def simulate_one():

    result = simulate(latent_dim=2,
             preprocess_type1=PreprocessType.STANDARDISATION_OVER_TENORS,
             preprocess_type2=PreprocessType.LOG_RETURNS_OVER_TENORS,
             ae_model=AEModel.AE,
             gan_model=GANModel.GAN,
             force_training=False,
             plot=True)

    print("result", np.mean(result), np.var(result))


if __name__ == '__main__':
    # simulate_one()
    simulate_many(force_training=True)  # `force_training=False` for testing only
