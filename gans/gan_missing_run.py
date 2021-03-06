from gans.gan_missing_model import GANMissing
from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
import numpy as np
import numpy.ma as ma
import hashlib
import json


def simulate():
    plotting = Plotting()
    preprocess_logreturns = PreprocessData()
    preprocess_logreturns.enable_log_returns = True

    # 1. get data and apply minimax
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess_logreturns.get_data()

    sets_training_first_last_tenors = []
    for set_training_scaled in sets_training_scaled:
        sets_training_first_last_tenors.append(set_training_scaled.iloc[:,[0,-1]])
    # sets_training_first_last_tenors = np.array(sets_training_first_last_tenors)

    sets_test_first_last_tenors = []
    for set_test_scaled in sets_test_scaled:
        sets_test_first_last_tenors.append(set_test_scaled.iloc[:,[0,-1]])
    # sets_test_first_last_tenors = np.array(sets_test_first_last_tenors)

    data_train = np.vstack(sets_training_first_last_tenors)
    mask = sample_mask(data_train.shape[0], data_train.shape[1], 0.2)
    data_masked = data_train * mask + (1-mask) * -1000 # missing values have representation -1000

    gan_params = {'num_tenors': sets_training_first_last_tenors[0].shape[1],
              'num_c': 6*7,
              'num_z': 6*7,
              'num_o': 6*7,
              'gen_model_type': 'standard', # conv
              'dis_model_type': 'standard', # conv
              'gen_layers': (4*(6*7*2),), # 4 * num_o * num_tenors
              'dis_layers': (4*(6*7),), # 4 * num_o
              'gen_last_activation': 'tanh',
              'dis_last_activation': 'sigmoid',
              'loss': 'binary_crossentropy',
              'batch_size': 128,
              'epochs': 20000,
              'mask_p': 0.2}
    gan_params_hash = hashlib.md5(json.dumps(gan_params, sort_keys=True).encode('utf-8')).hexdigest()

    gan = GANMissing(gan_params)
    gan.train(data_masked, mask)
    gan.save_model("gan_test_" + gan_params_hash)
    # gan.load_model("gan_test_" + gan_params_hash)

    # 4: simulate on encoded log returns, conditioned on test dataset
    def simulate_with_gan(num_runs=20):

        generated_segments, real_segment = gan.generate(data=sets_test_first_last_tenors[-1], num_simulations=100, remove_condition=False)
        last_generated_segment = generated_segments
        for _ in np.arange(num_runs - 1):
            generated_temp, real_temp = gan.generate(condition=last_generated_segment, remove_condition=True)
            last_generated_segment = generated_temp
            generated_segments = np.append(generated_segments, generated_temp, axis=1)

        # 5: undo log-returns
        generated_segments = preprocess_logreturns.rescale_data(generated_segments, start_value=sets_test[-1][-1])

        # plotting.plot_3d_many(file_name, data, save=False)
        plotting.plot_3d_training("3d recursively generated with GAN, test", generated_segments, sets_test[-1], show=True, after_real_data=True)

    for _ in np.arange(20):
        simulate_with_gan(num_runs=20)

def sample_mask(m, n, p):
    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1. * B
    return C

if __name__ == '__main__':
    simulate()
