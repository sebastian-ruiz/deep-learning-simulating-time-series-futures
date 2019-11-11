from autoencoders.autoencoder_model import Autoencoder
from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
from helpers.evaluate import *
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import json
from matplotlib.lines import Line2D


def simulate():
    plotting = Plotting()
    preprocess_type = PreprocessType.STANDARDISATION_OVER_TENORS
    preprocess = PreprocessData(preprocess_type)

    # 1. get data and apply minimax
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess.get_data()
    all_training_scaled = np.vstack(sets_training_scaled)

    ae_params = {'preprocess_type': preprocess_type.value,  # only to make preprocess_type part of the hash
                 'input_dim': sets_training_scaled[0].shape[1],  # 56
                 'latent_dim': 2,
                 'hidden_layers': (56, 40, 28, 12, 4,),
                 'leaky_relu': 0.1,
                 'loss': 'mse',
                 'last_activation': 'linear',
                 'batch_size': 20,
                 'epochs': 100,
                 'steps_per_epoch': 500}
    ae_params_hash = hashlib.md5(json.dumps(ae_params, sort_keys=True).encode('utf-8')).hexdigest()

    autoencoder = Autoencoder(ae_params)
    autoencoder.load_else_train(all_training_scaled, sets_test_scaled, "ae_" + ae_params_hash)

    # 2: encode data using autoencoder

    encoded = autoencoder.encode(sets_test_scaled[0])
    decoded = autoencoder.decode(encoded)

    rescaled = preprocess.rescale_data(decoded, dataset_name=test_dataset_names[0])
    smape_result = smape(rescaled, np.array(sets_test[0]), over_curves=True)

    print("smape_result test set", np.mean(smape_result), np.std(smape_result), np.min(smape_result), np.max(smape_result))

    plotting.plot_2d(sets_test[0], "evaluation of test curves", timeseries=True, evaluation=smape_result, title=False)


    # for i in np.arange(len(test_eval)):
    #     if test_eval[i] > 4:
    #         plotting.plot_2d(sets_test_scaled[0][i], "Possible unrealistic curve" + str(i), save=False, title=True)


    # 3: lets see how well the autoencoder can map a zero vector
    # todo: generate random curves, THEN apply min-max feature scaling, THEN evaluate
    unrealistic_curves = []
    curve_shape = 56
    unrealistic_curves.append(np.full(curve_shape, 5))
    unrealistic_curves.append(np.full(curve_shape, 10))
    unrealistic_curves.append(np.full(curve_shape, 20))
    unrealistic_curves.append(np.full(curve_shape, 50))
    unrealistic_curves.append(np.full(curve_shape, 70))
    unrealistic_curves.append(np.full(curve_shape, 100))
    unrealistic_curves.append(np.full(curve_shape, 150))
    unrealistic_curves.append(np.full(curve_shape, 200))
    unrealistic_curves.append(np.full(curve_shape, 250))
    unrealistic_curves.append(np.full(curve_shape, 300))
    unrealistic_curves.append(np.hstack((np.full(int(curve_shape / 2), 50), np.full(int(curve_shape / 2), 150))))
    unrealistic_curves.append(np.hstack((np.full(int(curve_shape/2), 100), np.full(int(curve_shape/2), 150))))
    unrealistic_curves.append(np.hstack((np.full(int(curve_shape / 2), 100), np.full(int(curve_shape / 2), 200))))
    unrealistic_curves.append(np.random.uniform(0,10, curve_shape))
    unrealistic_curves.append(np.random.uniform(10, 70, curve_shape))
    unrealistic_curves.append(np.random.uniform(0, 100, curve_shape))
    unrealistic_curves.append(np.random.uniform(100, 200, curve_shape))
    unrealistic_curves.append(np.random.uniform(200, 300, curve_shape))
    unrealistic_curves.append(np.random.uniform(0, 200, curve_shape))
    unrealistic_curves.append(np.random.uniform(0, 250, curve_shape))
    unrealistic_curves.append(np.random.uniform(0, 300, curve_shape))
    unrealistic_curves.append(np.linspace(0, 100, num=curve_shape))
    unrealistic_curves.append(np.linspace(50, 150, num=curve_shape))
    unrealistic_curves.append(np.linspace(100, 200, num=curve_shape))
    unrealistic_curves.append(np.linspace(150, 250, num=curve_shape))
    unrealistic_curves.append(np.linspace(200, 300, num=curve_shape))
    unrealistic_curves.append(np.linspace(0, 200, num=curve_shape))
    unrealistic_curves.append(np.linspace(0, 300, num=curve_shape))
    unrealistic_curves.append(np.linspace(100, 0, num=curve_shape))
    unrealistic_curves.append(np.linspace(150, 50, num=curve_shape))
    unrealistic_curves.append(np.linspace(200, 100, num=curve_shape))
    unrealistic_curves.append(np.linspace(250, 150, num=curve_shape))
    unrealistic_curves.append(np.linspace(300, 200, num=curve_shape))
    unrealistic_curves.append(np.linspace(200, 0, num=curve_shape))
    unrealistic_curves.append(np.linspace(300, 0, num=curve_shape))
    unrealistic_curves = np.array(unrealistic_curves)
    print("unrealistic_curves.shape", unrealistic_curves.shape)

    unrealistic_curves_scaled = preprocess.scale_data(unrealistic_curves, dataset_name=training_dataset_names[0], should_fit=True)

    encoded = autoencoder.encode(unrealistic_curves_scaled)
    decoded = autoencoder.decode(encoded)

    rescaled = preprocess.rescale_data(decoded, dataset_name=training_dataset_names[0])
    smape_result = smape(rescaled, unrealistic_curves, over_curves=True)

    round_to_n = lambda x, n: round(x, -int(np.floor(np.log10(x))) + (n - 1))

    print("smape results", smape_result)
    for a_smape_result in smape_result:
        print(round_to_n(a_smape_result, 2))

    plotting.plot_2d(smape_result, "loss of unrealistic curves from autoencoder SMAPE", save=False,
                     title=True)
    plotting.plot_2d(smape_result, "loss of unrealistic curves from autoencoder SMAPE", save=False, title=True)
    # plotting.plot_2d(unrealistic_eval_mse, "loss of unrealistic curves from autoencoder MSE", save=False, title=True)
    plotting.plot_unrealisticness(unrealistic_curves, "loss of unrealistic curves from autoencoder", timeseries=True, evaluation=smape_result, title=False, eval_label="SMAPE")

if __name__ == '__main__':
    simulate()
