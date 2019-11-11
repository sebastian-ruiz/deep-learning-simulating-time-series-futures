import numpy as np
import pandas as pd
from config import Config

if __name__ == '__main__':

    config = Config()

    data = pd.read_csv(
        "./data/GSPC.csv",
        header=1, parse_dates=True, index_col=[0],
        names=["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"]
    )

    data = data.drop(columns=["Open", "High", "Low", "AdjClose", "Volume"])

    print(data)
    data.to_pickle(config.get_filepath_encoded_data("gspc"))
