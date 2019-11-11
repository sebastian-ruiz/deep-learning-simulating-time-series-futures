from helpers.preprocess_data import PreprocessData
from helpers.plotting import Plotting
from config import Config
import matplotlib.pyplot as plt
import numpy as np

def main():

    preprocess_data = PreprocessData()
    plotting = Plotting()
    config = Config()

    preprocess_data.enable_min_max_scaler = True
    sets_training, sets_test, sets_training_scaled, sets_test_scaled, training_dataset_names, test_dataset_names, maturities = preprocess_data.get_data()

    wti_nymex = sets_test[0]
    print(wti_nymex)

    time = wti_nymex.axes[0].tolist()

    plotting.plot_3d("wti_nymex_time_series", wti_nymex, maturities, time)
    plotting.plot_some_curves("wti_nymex_some_curves", wti_nymex, curves=[0, 300, 600, 700], maturities=maturities)

    plotting.plot_2d(wti_nymex, "wti_nymex_short_vs_long_end", timeseries=True, save=True)

    plt.plot(wti_nymex.iloc[:, [0, -1]])
    plt.legend(["Short end", "Long end"])
    plt.ylabel("Price ($)")
    plt.xlabel("Time")
    plt.xticks(rotation=20)
    plt.savefig(config.get_filepath_img("wti_nymex_short_vs_long_end_old"), dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

    plotting.plotly_3d("wti_nymex_plotly", wti_nymex, maturities=maturities, time=time)

if __name__ == '__main__':
    main()