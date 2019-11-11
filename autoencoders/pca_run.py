import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
from autoencoders.pca_model import PCAModel
from sklearn.decomposition import PCA
from helpers.evaluate import *

def main():
    plotting = Plotting()
    preprocess_normalisation = PreprocessData()
    preprocess_normalisation.enable_normalisation_scaler = True
    # preprocess_normalisation.enable_standardisation_scaler = True

    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess_normalisation.get_data()

    # sklearn model (check that it is doing the same (it is))
    # pca_model_sklearn = PCA(n_components=2)
    # pca_model_sklearn.fit(sets_test_scaled[0])
    # test_data_scaled_encoded = pca_model_sklearn.transform(sets_test_scaled[0])
    # test_data_scaled_decoded = pca_model_sklearn.inverse_transform(test_data_scaled_encoded)

    # our own model
    def pca_on_normalised():

        params = {'latent_dim': 2}
        pca_model = PCAModel(params)
        pca_model.train(np.vstack(sets_training_scaled))

        test_data_scaled_encoded = pca_model.encode(sets_test_scaled[0])
        test_data_scaled_decoded = pca_model.decode(test_data_scaled_encoded)

        print("sets_test_scaled[0].shape", sets_test_scaled[0].shape)
        print("test_data_scaled_encoded.shape", test_data_scaled_encoded.shape)
        print("test_data_scaled_decoded.shape", test_data_scaled_decoded.shape)

        # plot results
        plotting.plot_2d(test_data_scaled_encoded, "wti_nymex_encoded_pca")
        simulated = preprocess_normalisation.rescale_data(test_data_scaled_decoded, dataset_name=test_dataset_names[0])
        plotting.plot_some_curves("wti_nymex_normalised_compare_pca", sets_test[0], simulated,
                                  [25, 50, 75, 815], maturities)

        # plotting.plot_some_curves("test_feature_normalised_compare_normalisation", sets_test[0], sets_test_scaled[0],
        #                           [25, 50, 75, 815, 100, 600, 720, 740], maturities, plot_separate=True)

        # print("reconstruction_error", reconstruction_error(sets_test_scaled[0], test_data_scaled_decoded))
        # print("reconstruction_error", reconstruction_error(np.array(sets_test[0]), simulated))

        print("smape", smape(np.array(sets_test[0]), simulated))
        # print("smape", np.mean(smape(np.array(sets_test[0]), simulated, over_curves=True)))


    def pca_on_unnormalised():
        pca_model = PCAModel(k=2)
        pca_model.train(np.vstack(sets_training))
        test_data_encoded = pca_model.encode(np.array(sets_test[0]))
        test_data_decoded = pca_model.decode(test_data_encoded)

        # plot results
        plotting.plot_2d(test_data_encoded.T, "wti_nymex_pca")
        # simulated = preprocess_normalisation.rescale_data(test_data_decoded, dataset_name=test_dataset_names[0])
        plotting.plot_some_curves("wti_nymex_compare_pca", sets_test[0], test_data_decoded,
                                  [25, 50, 75, 815], maturities)

    # pca_on_unnormalised()
    pca_on_normalised()



if __name__ == '__main__':
    main()
