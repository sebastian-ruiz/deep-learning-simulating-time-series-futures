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


def simulate(latent_dim=2, preprocess_type1=None, preprocess_type2=None, ae_model=None, plot=False):
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

    print("="*20)
    print(ae_model.name)
    print("\n")
    for set_encoded_log_training, training_dataset_name in zip(sets_encoded_log_training, training_dataset_names):
        print(training_dataset_name)
        print("min:", np.min(set_encoded_log_training.min()), "max:", np.max(set_encoded_log_training.max()))

    print("\n")

    for set_encoded_log_test, test_dataset_name in zip(sets_encoded_log_test, test_dataset_names):
        print(test_dataset_name)
        print("min:", np.min(set_encoded_log_test.min()), "max:", np.max(set_encoded_log_test.max()))

    print("\n")
    print("=" * 20)


def simulate_many():

    for preprocess_type in [PreprocessType.STANDARDISATION_OVER_TENORS]:
        for ae_model in [AEModel.PCA, AEModel.AE, AEModel.AAE]:
            for hidden_dim in [2]:
                simulate(hidden_dim, preprocess_type, PreprocessType.LOG_RETURNS_OVER_TENORS, ae_model)


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


if __name__ == '__main__':
    simulate_many()
