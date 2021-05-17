import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def read_data(input_file):
    input_data = np.loadtxt(input_file, delimiter = None)
    dates = pd.date_range('1950-01', periods=input_data.shape[0], freq='M')
    output = pd.Series(input_data[:, index], index=dates)
    return output


if __name__ == '__main__':
    input_file = "C:/Users/User/Desktop/ao.txt"
    timeseries = read_data(input_file)
    timeseries.describe()
    plt.figure()
    timeseries.rolling(window=12, center=False).mean().plot(style='-g')
    plt.show()