import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv('indian_pines.csv')
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
data = df.iloc[:, :-1] .to_numpy()

for idx in [0,100,200]:
    signal = data[idx]
    first_derivative = np.gradient(signal)
    second_derivative = np.gradient(first_derivative)
    abs_second_derivative = np.abs(second_derivative)
    abs_first_derivative = np.abs(first_derivative)
    d = np.diff(signal)
    ad = np.abs(d)
    print(d.shape)

    # plt.plot(signal)
    # plt.show()
    #
    # plt.plot(first_derivative)
    # plt.show()

    # plt.plot(d)
    # plt.show()
    #
    # i = list(range(len(abs_first_derivative)))
    # plt.bar(i,abs_first_derivative)
    # plt.show()

    i = list(range(len(ad)))
    plt.bar(i,ad)
    plt.show()

    # plt.plot(second_derivative)
    # plt.show()
    #
    # i = list(range(len(abs_second_derivative)))
    # plt.bar(i,abs_second_derivative)
    # plt.show()

