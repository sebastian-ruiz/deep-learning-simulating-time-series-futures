from gans.cwgan_gp_model import CWGANGP
from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
from sklearn.preprocessing import MinMaxScaler
import numpy as np
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
        sets_training_first_last_tenors.append(set_training_scaled[:,[0,-1]])
    # sets_training_first_last_tenors = np.array(sets_training_first_last_tenors)

    sets_test_first_last_tenors = []
    for set_test_scaled in sets_test_scaled:
        sets_test_first_last_tenors.append(set_test_scaled[:,[0,-1]])
    # sets_test_first_last_tenors = np.array(sets_test_first_last_tenors)

    # scaler = MinMaxScaler()
    to_train = np.vstack(sets_training_first_last_tenors)
    to_train = scaler.fit_transform(np.vstack(sets_training_first_last_tenors))

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
              'batch_size': 32,
              'epochs': 10000,
              'sample_interval': 1000}
    gan_params_hash = hashlib.md5(json.dumps(gan_params, sort_keys=True).encode('utf-8')).hexdigest()

    # print("min, max", np.min(np.vstack(sets_training_first_last_tenors)), np.max(np.vstack(sets_training_first_last_tenors)))

    gan = CWGANGP(gan_params)
    gan.train(to_train)
    gan.save_model("gwgan_gp_" + gan_params_hash)
    # gan.load_model("gwgan_gp_" + gan_params_hash)

    # 4: simulate on encoded log returns, conditioned on test dataset
    num_simulations = 10
    num_repeats = 20
    generated_segments, real_segment = gan.generate(data=sets_test_first_last_tenors[-1], num_simulations=num_simulations, remove_condition=False)
    last_generated_segment = generated_segments
    for _ in np.arange(num_repeats - 1):
        generated_temp, real_temp = gan.generate(condition=last_generated_segment, remove_condition=True)
        last_generated_segment = generated_temp
        generated_segments = np.append(generated_segments, generated_temp, axis=1)

    # undo scaler:
    # temp_generated_segments = []
    # for i in np.arange(generated_segments.shape[0]):
    #     temp_generated_segments.append(scaler.inverse_transform(generated_segments[i]))
    # generated_segments = np.array(temp_generated_segments)
    # 5: undo log-returns
    generated_segments = preprocess_logreturns.rescale_data(generated_segments, start_value=sets_test_first_last_tenors[-1][-1])
    # plotting.plot_3d_many(file_name, data, save=False)
    plotting.plot_3d_training("3d recursively generated with GAN, test", generated_segments, sets_test[-1], show=True, after_real_data=True)

if __name__ == '__main__':
    simulate()
