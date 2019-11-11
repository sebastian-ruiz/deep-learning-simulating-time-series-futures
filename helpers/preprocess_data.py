import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
import numpy as np
from pathlib import Path
from config import Config
from helpers.plotting import Plotting
from helpers.evaluate import PreprocessType
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import yaml
from scipy.signal import savgol_filter
import numbers

class PreprocessData:
    def __init__(self, preprocess_type=None, extend_data=False, short_end=False):

        self.config = Config()
        # prepare input data
        config_path = self.config.get_filepath("", "config.yaml")

        config_file = open(config_path, 'r')
        yaml_config = yaml.load(config_file, Loader=yaml.SafeLoader)

        self.training_dataset_names = [d['name'] for d in yaml_config['training_datasets']]
        self.training_dataset_start_pos = [d['start_position'] for d in yaml_config['training_datasets']]
        self.test_dataset_names = [d['name'] for d in yaml_config['test_datasets']]
        self.test_dataset_start_pos = [d['start_position'] for d in yaml_config['test_datasets']]
        self.dataset_names = np.concatenate((self.training_dataset_names, self.test_dataset_names)) # do we need these?
        self.dataset_start_pos = np.concatenate((self.training_dataset_start_pos, self.test_dataset_start_pos)) # do we need these?

        # read in all pickle files
        self.all_pd = []
        for dataset_name in self.dataset_names:
            self.all_pd.append(pd.read_pickle(self.config.get_filepath_data(dataset_name)))

        if extend_data:
            training_dataset_names_copy = np.array(self.training_dataset_names, copy=True)

            # create a copy of the data shifted up by 10
            for i, dataset_name in enumerate(training_dataset_names_copy):
                self.dataset_names = np.append(self.dataset_names, dataset_name + "_" + str(10))
                self.training_dataset_names = np.append(self.training_dataset_names, dataset_name + "_" + str(10))
                self.dataset_start_pos = np.append(self.dataset_start_pos, self.training_dataset_start_pos[i])
                self.training_dataset_start_pos.append(self.training_dataset_start_pos[i])
                self.all_pd.append(self.all_pd[i].copy() + 10)

        self.dict_datasets = dict(zip(self.dataset_names, np.arange(len(self.dataset_names))))

        self.enable_difference = False

        self._feature_range = [0, 1]
        self.normalisation_scalers = []
        for _ in self.dataset_names:
            self.normalisation_scalers.append(MinMaxScaler(feature_range=self.feature_range))

        self.enable_normalisation_scaler = False
        self.enable_ignore_price = False # scale each curve to feature_range

        self.power_transformer = PowerTransformer()
        self.enable_power_transform = False

        self.standardisation_scalers = []
        for _ in self.dataset_names:
            self.standardisation_scalers.append(StandardScaler())

        self.enable_standardisation_scaler = False

        self.enable_log_returns = False
        self.mult_factor = 10  # 5
        self.add_factor = 25  # 6

        self.enable_log = False
        self.enable_pct_change = False

        self.enable_curve_smoothing = False

        self.short_end = short_end

        # now setup PreprocessType settings
        if preprocess_type is PreprocessType.NORMALISATION_OVER_TENORS:
            self.enable_normalisation_scaler = True
            self.feature_range = [0, 1]
        elif preprocess_type is PreprocessType.NORMALISATION_OVER_CURVES:
            self.enable_normalisation_scaler = True
            self.feature_range = [0, 1]
            self.enable_ignore_price = True
        elif preprocess_type is PreprocessType.STANDARDISATION_OVER_TENORS:
            self.enable_standardisation_scaler = True
        elif preprocess_type is PreprocessType.LOG_RETURNS_OVER_TENORS:
            self.enable_log_returns = True

    @property
    def feature_range(self):  # implements the get - this name is *the* name
        return self._feature_range

    @feature_range.setter
    def feature_range(self, value):  # name must be the same
        self._feature_range = value
        for i, _ in enumerate(self.dataset_names):
            self.normalisation_scalers[i] = MinMaxScaler(feature_range=value)

    def get_data(self, training_dataset_names=None, test_dataset_names=None, chunks_of=None):

        if training_dataset_names is None:
            training_dataset_names = self.training_dataset_names
        if isinstance(training_dataset_names, str):
            training_dataset_names = np.array([training_dataset_names])

        if test_dataset_names is None:
            test_dataset_names = self.test_dataset_names
        if test_dataset_names is None and self.test_dataset_names is None:
            test_dataset_names = []

        if isinstance(test_dataset_names, str):
            test_dataset_names = np.array([test_dataset_names])

        training_data = []
        test_data = []
        training_data_scaled = []
        test_data_scaled = []
        for key, value in self.dict_datasets.items():
            start_position = self.dataset_start_pos[value]
            end_position = None
            if chunks_of is not None:
                end_position = chunks_of * ((self.all_pd[value].shape[0] - start_position) // chunks_of)

            if key in training_dataset_names:
                # we take the log returns of each data set and scale wrt first dataset
                new_training_data = self.all_pd[value].copy()[start_position:end_position]
                if self.short_end:
                    new_training_data = new_training_data.iloc[:, 0]

                new_training_data_scaled = self.scale_data(new_training_data, value, True)

                training_data.append(new_training_data)
                training_data_scaled.append(new_training_data_scaled)

            if key in test_dataset_names:
                new_test_data = self.all_pd[value].copy()[start_position:end_position]
                if self.short_end:
                    new_test_data = new_test_data.iloc[:, 0]

                new_test_data_scaled = self.scale_data(new_test_data, value, True) # todo: should we scale test data wrt training data?

                test_data.append(new_test_data)
                test_data_scaled.append(new_test_data_scaled)

        maturities = self.all_pd[0].columns.values/(30 * 12) # for years

        if test_dataset_names is not None:
            return training_data, test_data, training_data_scaled, test_data_scaled, training_dataset_names, test_dataset_names, maturities
        else:
            return training_data_scaled, maturities

    # def rescale_data_inputter(self, data, datasets=None):
    #     rescaled_data = []
    #     if datasets == "train":
    #         for i, name in enumerate(self.training_dataset_names):
    #             # pos = self.dict_datasets[name]
    #             rescaled_data.append(self.rescale_data(data[i], dataset_name=name))
    #
    #     elif datasets == "test":
    #         for i, name in enumerate(self.test_dataset_names):
    #             # pos = self.dict_datasets[name]
    #             # self.scale_data(self, data, dataset_num=pos)
    #             rescaled_data.append(self.rescale_data(data[i], dataset_name=name))
    #
    #     return rescaled_data

    def scale_data(self, data, dataset_name=None, should_fit=False):

        # if given a numpy array, convert it to a dataframe first
        if type(data) is np.ndarray:
            _data = pd.DataFrame(data=data)
        elif isinstance(data, list):
            _data_list = []
            # if isinstance(dataset_name, list):
            for _data, _dataset_name in zip(data, dataset_name):
                _data_list.append(self.scale_data(_data, _dataset_name, should_fit))
            # else:
            #     for _data in data:
            #         _data_list.append(self.scale_data(_data, should_fit, dataset_name))
            return _data_list
        else:
            _data = data.copy()

        time = _data.axes[0].tolist()
        # maturities = _data.columns.values

        dataset_num = 999
        if dataset_name is not None:
            if isinstance(dataset_name, numbers.Integral):
                dataset_num = dataset_name
            else:
                for key, value in self.dict_datasets.items():
                    if key == dataset_name:
                        dataset_num = value

        if self.enable_log:
            _data = _data.apply(np.log)

        if self.enable_difference:
            _data = _data.diff(axis=1)
            _data = _data.fillna(0)

        if self.enable_pct_change:
            _data = _data.pct_change()
            _data = _data.fillna(0)

        if self.enable_log_returns:
            shift = (_data.shift(0) + self.add_factor) / (
                        _data.shift(1) + self.add_factor)  # add 6 to make it non-negative, to take the log later
            shift = shift.dropna()

            if not (np.array(shift) > 0).all():
                # some values are non-positive... this will break the log
                print("NON-POSITIVE VALUES FOUND, CANNOT PASS THROUGH LOG!!")
                print(np.min(_data))
                print(shift)

            _data = self.mult_factor * np.log(shift)

            time = _data.axes[0].tolist()

        # now use only numpy, convert pandas to numpy array
        _data = _data.values

        if self.short_end and len(_data.shape) == 1:
            _data = _data.reshape(-1, 1)

        if self.enable_standardisation_scaler:
            if not self.enable_ignore_price:
                if should_fit:
                    self.standardisation_scalers[dataset_num].fit(_data)
                _data = self.standardisation_scalers[dataset_num].transform(_data)
            else:
                data_temp = []
                for row in _data:
                    # row_as_2d = row.reshape(1, -1)
                    row_as_column = row[:, np.newaxis]
                    self.standardisation_scalers[dataset_num].fit(row_as_column)
                    temp = self.standardisation_scalers[dataset_num].transform(row_as_column)
                    data_temp.append(temp.ravel())
                _data = np.array(data_temp)

        if self.enable_normalisation_scaler:
            if not self.enable_ignore_price:
                if should_fit:
                    self.normalisation_scalers[dataset_num].fit(_data)
                _data = self.normalisation_scalers[dataset_num].transform(_data)
            else:
                data_temp = []
                for row in _data:
                    # row_as_2d = row.reshape(1, -1)
                    row_as_column = row[:, np.newaxis]
                    self.normalisation_scalers[dataset_num].fit(row_as_column)
                    temp = self.normalisation_scalers[dataset_num].transform(row_as_column)
                    data_temp.append(temp.ravel())
                _data = np.array(data_temp)

        if self.enable_power_transform:
            if should_fit:
                self.power_transformer.fit(_data)
            _data = self.power_transformer.transform(_data)

        df = pd.DataFrame(data=_data, index=np.array(time))

        return df

    def rescale_data(self, data, dataset_name=None, start_value=None, index=None, columns=None):

        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.columns.values
            if index is None:
                index = data.index.values

        if type(data) is np.ndarray:
            temp_data = data
        else:
            temp_data = np.array(data)

        if self.short_end and len(temp_data.shape) == 1:
            temp_data = temp_data.reshape(-1, 1)

        dataset_num = 999
        if dataset_name is not None:
            for key, value in self.dict_datasets.items():
                if key == dataset_name:
                    dataset_num = value

        if self.enable_difference:
            temp_data = temp_data  # TODO: inverse difference

        if self.enable_power_transform:
            temp_data = self.power_transformer.inverse_transform(temp_data)

        if self.enable_normalisation_scaler:

            # we need to scale each rolling window manually
            if self.enable_ignore_price:
                # rescale each curve individually
                data_min = self.all_pd[dataset_num].min(axis=1)
                data_max = self.all_pd[dataset_num].max(axis=1)
                a = self.feature_range[0]
                b = self.feature_range[1]
                for i in np.arange(temp_data.shape[0]):
                    temp_data[i] = ((temp_data[i] - a)/(b-a)) * (data_max[i] - data_min[i]) + data_min[i]
            else:
                if len(temp_data.shape) == 3:
                    new_temp_data = []
                    for i in np.arange(temp_data.shape[0]):
                        new_temp_data.append(self.normalisation_scalers[dataset_num].inverse_transform(temp_data[i]))
                    temp_data = np.array(new_temp_data)

                else:
                    temp_data = self.normalisation_scalers[dataset_num].inverse_transform(temp_data)

        if self.enable_standardisation_scaler:
            # temp_data = self.standardisation_scaler.inverse_transform(temp_data)
            if self.enable_ignore_price:
                raise NotImplementedError
            else:
                if len(temp_data.shape) == 3:
                    new_temp_data = []
                    for i in np.arange(temp_data.shape[0]):
                        new_temp_data.append(self.standardisation_scalers[dataset_num].inverse_transform(temp_data[i]))
                    temp_data = np.array(new_temp_data)

                else:
                    temp_data = self.standardisation_scalers[dataset_num].inverse_transform(temp_data)


        if self.enable_log:
            temp_data = np.exp(temp_data)

        if self.enable_log_returns:

            # if start_value is not assigned but dataset_name is, use the first value of the dataset as start_value
            if dataset_name is not None and start_value is None:
                _start_value = self.all_pd[dataset_num].iloc[0]
            elif start_value is not None:
                _start_value = start_value
            else:
                _start_value = 1.

            # print("shapes, log-return rescale", temp_data.shape, _start_value.shape, _start_value[0].shape)

            if len(temp_data.shape) is 1:
                z = np.exp(temp_data / self.mult_factor)

                z = np.insert(np.array(z), 0, _start_value[0] + self.add_factor) # instead of the usual _start_value
                temp_data = np.cumprod(z) - self.add_factor
                temp_data = pd.DataFrame(data=temp_data, index=self.all_pd[dataset_num].index)
                # print(temp_data.head(10))
            elif len(temp_data.shape) is 2: # when taking log-returns on an individual batch, todo: check

                if self.short_end:
                    z = np.exp(temp_data / self.mult_factor)
                    z = np.insert(z, 0, _start_value[0] + self.add_factor, axis=0)
                    temp_data = np.cumprod(z, axis=0) - self.add_factor
                else:
                    z = np.exp(temp_data / self.mult_factor)
                    z = np.insert(z, 0, _start_value + self.add_factor, axis=0)
                    temp_data = np.cumprod(z, axis=0) - self.add_factor

            elif len(temp_data.shape) > 2: # when taking log-returns on multiple batches
                z = np.exp(temp_data[:, :] / self.mult_factor)
                z = np.insert(z, 0, _start_value + self.add_factor, axis=1)
                temp_data = np.cumprod(z, axis=1) - self.add_factor
            else:
                z = np.exp(temp_data[0, :] / self.mult_factor)
                z = np.insert(z, 0, _start_value + self.add_factor)
                temp_data = np.cumprod(z) - self.add_factor

            # print("log returns undo...", _start_value, temp_data[0])

        if self.enable_curve_smoothing:
            curve_smooth = []

            for curve in temp_data:
                curve_smooth.append(savgol_filter(curve, 23, 5))  # window size 51, polynomial order 3
            temp_data = np.array(curve_smooth)

        if index is not None and columns is not None:
            return pd.DataFrame(temp_data, index=index, columns=columns)
        else:
            return temp_data
