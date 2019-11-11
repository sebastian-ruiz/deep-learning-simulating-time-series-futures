from helpers.preprocess_data import PreprocessData
from helpers.evaluate import *
from helpers.plotting import Plotting
from imputance.gain_model import gain
import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':
    plotting = Plotting()
    preprocess = PreprocessData(PreprocessType.STANDARDISATION_OVER_TENORS, short_end=True)
    preprocess2 = PreprocessData(PreprocessType.LOG_RETURNS_OVER_TENORS, short_end=True)
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess.get_data()

    sets_encoded_log_training = preprocess2.scale_data(sets_training_scaled, training_dataset_names, should_fit=True)
    sets_encoded_log_test = preprocess2.scale_data(sets_test_scaled, test_dataset_names, should_fit=True)

    train = sets_encoded_log_training[0].copy()
    test = sets_encoded_log_test[0].copy()

    # print("train.shape[1]", train.shape[1])
    # print("sets_test_scaled[0]", sets_test_scaled[0].shape)
    # print("sets_encoded_log_test[0]", sets_encoded_log_test[0].shape)




    params = {
        'mb_size': 128,  # 'mb_size': 128,
        'p_miss': 0.5,  #  'p_miss': 0.5, doesn't do anything
        'p_hint': 0.9,  # 'p_hint': 0.9
        'alpha': 10,  # 'alpha': 10,
        'dim': train.shape[1],
        'train_no': len(train),
        'test_no': len(test),
        'epochs': 10000,
    }

    gain = gain(params)
    test_mask = gain.make_mask(test)

    # print(test.head(10))

    print("test_mask.shape", test_mask.shape)
    print("test.shape", test.shape)
    print("sets_test[0].shape", sets_test[0].shape)

    test_with_mask = test * test_mask

    # standardised_test_with_mask = preprocess2.rescale_data(test_with_mask, test_dataset_names[0], start_value=sets_test_scaled[0][0], index=sets_test_scaled[0].index.values)
    # rescaled_test_with_mask = preprocess.rescale_data(standardised_test_with_mask, test_dataset_names[0])

    # plotting.plot_2d(test_with_mask.mask(test_with_mask == 0), "gain_test_masked", title=True)
    # plotting.plot_2d(test_with_mask, "gain_test_masked", title=True)

    # plotting.plot_2d(rescaled_test_with_mask, "test_set_masked", title=True)

    gain.build_model()
    gain.train(sets_encoded_log_training, test, test_mask)

    test_prediction = gain.predict(test, test_mask)

    print("test.head(10)", test.head(10))
    print("test_prediction.head(10)", test_prediction.head(10))

    standardised_test_prediction = preprocess2.rescale_data(test_prediction, test_dataset_names[0], start_value=sets_test_scaled[0][0], index=sets_test_scaled[0].index.values)
    rescaled_test_prediction = preprocess.rescale_data(standardised_test_prediction, test_dataset_names[0])

    # print("isinstance(rescaled_test_prediction, pd.DataFrame)", isinstance(rescaled_test_prediction, pd.DataFrame))

    plotting.plot_2d(sets_test[0], "sets_test[0]", title=True)
    plotting.plot_2d(standardised_test_prediction, "standardised_test_prediction", title=True)
    plotting.plot_2d(rescaled_test_prediction, "rescaled_test_prediction", title=True)
    # plotting.plot_2d(rescaled_test_with_mask, "rescaled_test_with_mask", title=True)
    plotting.plot_2d(rescaled_test_prediction, "rescaled_test_prediction", title=True)

    # plotting.plot_2d(test, "gain_test_prediction", curve2=test_prediction, title=True)
