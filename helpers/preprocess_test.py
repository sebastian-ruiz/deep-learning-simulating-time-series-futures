import unittest
import numpy as np
from helpers.preprocess_data import PreprocessData
from helpers.evaluate import *

class PreprocessTest(unittest.TestCase):

    def helper(self, preprocess_type):
        preprocess = PreprocessData()
        if preprocess_type is None or preprocess_type is PreprocessType.NORMALISATION_OVER_TENORS:
            preprocess.enable_normalisation_scaler = True
            preprocess.feature_range = [0, 1]
        elif preprocess_type is PreprocessType.NORMALISATION_OVER_CURVES:
            preprocess.enable_normalisation_scaler = True
            preprocess.feature_range = [0, 1]
            preprocess.enable_ignore_price = True
        elif preprocess_type is PreprocessType.STANDARDISATION_OVER_TENORS:
            preprocess.enable_standardisation_scaler = True
        elif preprocess_type is PreprocessType.LOG_RETURNS_OVER_TENORS:
            preprocess.enable_log_returns = True

        sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess.get_data()
        rescaled_first_test_set = preprocess.rescale_data(sets_test_scaled[0], test_dataset_names[0])

        # check that assert_allclose is working:
        # rand = np.random.random_sample(sets_test[0].shape)
        # np.testing.assert_allclose(rescaled_first_test_set, rand)

        np.testing.assert_allclose(rescaled_first_test_set, sets_test[0])

    def test_normalisation_over_tenors(self):
        self.helper(PreprocessType.NORMALISATION_OVER_TENORS)

    def test_normalisation_over_curves(self):
        self.helper(PreprocessType.NORMALISATION_OVER_CURVES)

    def test_standardisation_over_tenors(self):
        self.helper(PreprocessType.STANDARDISATION_OVER_TENORS)

    def test_log_returns(self):
        self.helper(PreprocessType.LOG_RETURNS_OVER_TENORS)

    def test_two_preprocessing_methods(self):
        preprocess = PreprocessData(PreprocessType.STANDARDISATION_OVER_TENORS, short_end=True)
        preprocess2 = PreprocessData(PreprocessType.LOG_RETURNS_OVER_TENORS, short_end=True)
        sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess.get_data()

        sets_encoded_log_test = preprocess2.scale_data(sets_test_scaled, test_dataset_names, should_fit=True)

        # in this case the start_value is required, otherwise it will take the start_value of the original data instead
        standardised_test_prediction = preprocess2.rescale_data(sets_encoded_log_test[0], test_dataset_names[0],
                                                                start_value=sets_test_scaled[0][0],
                                                                index=sets_test_scaled[0].index.values)
        rescaled_test_prediction = preprocess.rescale_data(standardised_test_prediction, test_dataset_names[0])

        # plotting.plot_2d(sets_test[0], "gain_test_prediction_rescaled", curve2=rescaled_test_prediction, title=True)

        np.testing.assert_allclose(rescaled_test_prediction, sets_test[0])

if __name__ == '__main__':
    unittest.main()
