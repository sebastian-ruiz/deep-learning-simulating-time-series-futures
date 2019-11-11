import logging
import sys
import yaml
import os
import datetime
import inspect
from pathlib import Path
from calibrationlib import *
from helpers import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.dates as pltdates
from plotly.offline import plot
import plotly.graph_objs as go
from config import Config

class DataImporter():

    def __init__(self):

        # self.commodities = ["ETH CONWAY IN E-P USD-GAL", "IBUT CONWAY USD-GAL", "WTI NYMEX", "BR ICE"]
        self.config = Config()

    def import_commodities(self):

        input_folder = ''
        working_folder = self.config.get_path_caches()
        config_path = None
        base_date = ''
        output_file = ''
        crefile = '',
        md_file = ''
        n_cores = ''
        injection = ''

        if not config_path:
            config_path = self.config.get_filepath("/data_importer", "config.yaml")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = dict(yaml.load(f, Loader=yaml.FullLoader).items())
                if injection:
                    exec(injection)
        except Exception as e:
            print('Error: cannot read config file {}.'.format(config_path))
            raise Exception('Failed to read config file {}.'.format(config_path)) from e

        if not working_folder:
            working_folder = config['writableDir']

        try:
            intermediate_folder = config['intermediateFolder'].format(working_folder)
        except Exception as e:
            print('Error: cannot find intermediate folder name in config file {}.'.format(config_path))
            raise Exception('Failed to read config file {}.'.format(config_path)) from e

        try:
            os.makedirs(intermediate_folder, exist_ok=True)
        except Exception as e:
            print('Error: cannot create intermediate folder {}.'.format(intermediate_folder))
            raise Exception('Failed to create intermediate folder {}.'.format(intermediate_folder)) from e

        log_path = 'calibrate_{0}_{1}.log'.format(base_date, datetime.date.today().strftime('%d%b%Y'))
        log_path = os.path.join(intermediate_folder, log_path)
        try:
            logging.basicConfig(filename=log_path, level=logging.INFO)
        except Exception as e:
            print('Error: cannot open log file {}.'.format(log_path))
            raise Exception('Failed to open log file {}.'.format(log_path)) from e


        try:
            if not input_folder:
                input_folder = config['dataSource0']

            if not base_date:
                base_date = config['baseDate']

            if not output_file:
                output_file = config['outputFile']
            else:
                output_file = os.path.join(working_folder, output_file)

        except:
            logging.exception('Error reading config file {}.'.format(config_path))
            sys.exit(1)

        harv = Harvester(config, base_date, intermediate_folder, input_folder)
        major_commodities = config['majorCommodities']

        for commodity in major_commodities:
            commodity_name = commodity['calibrationCurve']
            print(commodity_name.replace(" ", "_").lower())

        for commodity in major_commodities:
            commodity_name = commodity['calibrationCurve'] # calibrationCurve
            # print("importing " + commodity_name)
            harvest_data = harv.get(commodity_name)

            if commodity_name == "WTI NYMEX":
                print("WTI NYMEX")

            self.save_to_pickle(harvest_data, commodity_name)
            # print("imported " + commodity_name + "\n")

    def save_to_pickle(self, harvest_data, commodity_name):

        df = pd.DataFrame.from_dict(harvest_data, orient='index')
        df.reset_index(inplace=True)  # make date a column
        df.drop(columns=[1], inplace=True, errors='ignore')  # drop duplicate column
        df.rename(index=str, columns={"index": "date", 0: "curve"}, inplace=True)
        df.sort_values(by='date', inplace=True)  # sort by date

        begin_date = df["date"].iloc[0]
        end_date = df["date"].iloc[-1]

        print("\n")
        print("="*30)
        print("commodity_name:", commodity_name)
        print("first_date:", begin_date)
        print("end_date:", end_date)

        # BEGIN ========================================================================================================
        # compute maximum date to plot to
        # take first_date as our starting point first_curve.x[-1] is the last x value point on the first curve

        # end_of_x_range = []
        # for index, row in df.iterrows():
        #     curve = row["curve"]
        #     # print(curve)
        #     end_of_x_range.append(curve.x[-1])
        #     # print(curve.x[-1])
        #
        # # for br_ice this is around 1800
        # max_x_values = np.amax(np.asarray(end_of_x_range))
        # print("max x values:", max_x_values)
        # END. =========================================================================================================

        # we choose 1680 because then we get 56 = 2^3 * 7 features. We want it to be divisible by two a lot to use CNN.
        x_labels = np.arange(start=0, stop=1680, step=30)

        def get_interp_values(a_curve):
            if a_curve.x[0] < 0 or a_curve.x[0] > 30:
                print("ERROR: a_curve.x[0]=", a_curve.x[0])

            return pd.Series(a_curve(x_labels))

        df[x_labels] = df['curve'].apply(get_interp_values)

        df.drop(columns=['curve'], inplace=True)
        df.set_index('date', inplace=True)

        print(df.head(5))

        df.to_pickle(self.config.get_filepath_data(commodity_name))


if __name__ == '__main__':
    data_importer = DataImporter()
    data_importer.import_commodities()
