from gans.gan_3d_simple_model import GAN
from autoencoders.deep_autoencoder_model import DeepAutoencoder
from autoencoders.adversarial_autoencoder_model import AdversarialAutoencoder
from helpers.encoded_data import EncodedData
from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
import numpy as np
import matplotlib.pyplot as plt

def simulate():
    plotting = Plotting()
    preprocess_minmax = PreprocessData()
    preprocess_logreturns = PreprocessData()
    preprocess_minmax.enable_min_max_scaler = True
    preprocess_logreturns.enable_log_returns = True

    # 1. get data and apply minimax
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess_minmax.get_data()

    print("sets_training_scaled.shape", sets_training_scaled[0].shape)

    autoencoder = DeepAutoencoder(input_shape=(sets_training_scaled[0].shape[1],), latent_dim=2)
    # autoencoder.train(np.vstack(sets_training_scaled), sets_test_scaled, epochs=100, batch_size=5)
    # autoencoder.save_model("deep_general_minimax")
    autoencoder.load_model("deep_general_minimax")

    # 2: encode data using autoencoder
    sets_encoded_training = []
    for set_training_scaled in sets_training_scaled:
        sets_encoded_training.append(autoencoder.encode(set_training_scaled))

    sets_encoded_test = []
    for set_test_scaled in sets_test_scaled:
        sets_encoded_test.append(autoencoder.encode(set_test_scaled))

    plotting.plot_2d(sets_encoded_test[0], "encoded test data with deep autoencoder", save=False)


    # 3: log returns of encoded data
    sets_encoded_log_training = []
    for index, set_encoded_training in enumerate(sets_encoded_training):
        sets_encoded_log_training.append(preprocess_logreturns.scale_data(set_encoded_training))

    sets_encoded_log_test = []
    for index, set_encoded_test in enumerate(sets_encoded_test):
        sets_encoded_log_test.append(preprocess_logreturns.scale_data(set_encoded_test))

    plotting.plot_2d(sets_encoded_log_test[0], "encoded test data with deep autoencoder, then log returns", save=False)


    num_tenors = sets_encoded_log_training[0].shape[1]
    gan = GAN(num_c=6 * 7, num_z=6 * 7, num_o=6 * 7, num_tenors=num_tenors)  # try training on larger input and output
    # gan.train(sets_encoded_log_training, epochs=20000, batch_size=100, sample_interval=200)
    # gan.save_model("general_ae")
    gan.load_model("general_ae")

    print("sets_encoded_log_test[0].shape", sets_encoded_log_test[0].shape)

    test_arr = np.full([1, 6*7 + 6*7, num_tenors], 10)

    validity = gan.discriminator.predict(test_arr) # np.array(sets_encoded_log_test[0]
    print(validity)

    rolled_encoded_log_test = rolling_windows(sets_encoded_log_test[0], 6*7 + 6*7)

    validity = gan.discriminator.predict(rolled_encoded_log_test)  # np.array(sets_encoded_log_test[0]
    print(validity)

    # 4: simulate on encoded log returns, conditioned on test dataset
    # def simulate_with_gan(num_runs=20):
    #
    #     generated_segments, real_segment = gan.generate(data=sets_encoded_log_test[-1], num_simulations=100, remove_condition=False)
    #     last_generated_segment = generated_segments
    #     for _ in np.arange(num_runs - 1):
    #         generated_temp, real_temp = gan.generate(condition=last_generated_segment, remove_condition=True)
    #         last_generated_segment = generated_temp
    #         generated_segments = np.append(generated_segments, generated_temp, axis=1)
    #
    #     # 5: undo log-returns
    #     encoded_generated_segments = preprocess_logreturns.rescale_data(generated_segments, start_value=sets_encoded_test[-1][-1])
    #     # 6: decode using autoencoder
    #     decoded_generated_segments = autoencoder.decode(encoded_generated_segments)
    #
    #     # 7: undo minimax, for now only the first simulation
    #     decoded_generated_segments_first_sim = decoded_generated_segments[0]
    #     simulated = preprocess_minmax.rescale_data(decoded_generated_segments_first_sim, dataset_name=test_dataset_names[-1])
    #
    #     # plotting.plot_3d_many(file_name, data, save=False)
    #
    #     # plotting.plot_3d_training("3d recursively generated with GAN", generated_segments, real_segment, show=True)
    #     plotting.plot_3d_training("autoencoded 3d recursively generated with GAN", simulated, sets_test[-1], show=True, after_real_data=True)
    #     # plotting.plotly_3d("autoencoded 3d recursively generated with GAN", simulated, sets_test[-1])
    #
    #
    # for _ in np.arange(20):
    #     simulate_with_gan(num_runs=20)

def rolling_windows(data, pattern_len, disjoint=True):

    n = data.shape[0] - pattern_len
    if disjoint:
        indices = np.arange(0, n, pattern_len)
    else:
        indices = np.arange(0, n)

    print("pattern_len", pattern_len, "n", n)
    print("data.shape", data.shape)
    print("indices", indices)

    out = np.array([data[a:a+pattern_len, :] for a in indices])

    print("out.shape", out.shape)

    return out

if __name__ == '__main__':
    simulate()
