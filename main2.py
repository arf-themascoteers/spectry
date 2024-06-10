import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import train_test_evaluator

def get_diff(X):
    diff = np.diff(X, axis=1)
    adiff = np.abs(diff)
    adiff_padded = np.concatenate((adiff[:, 0:1], adiff, adiff[:, -1:]), axis=1)
    adiff_mean = (adiff_padded[:, :-1] + adiff_padded[:, 1:]) / 2
    return adiff_mean

df = pd.read_csv('indian_pines.csv')
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
data = df.to_numpy()
train, test = train_test_split(data, test_size=0.2, random_state=1)
oa, aa, k = train_test_evaluator.evaluate_train_test_pair(train[:,:-1], train[:,-1], test[:,:-1], test[:,-1])
print(oa, aa, k)

train_x = get_diff(train[:,:-1])
test_x = get_diff(test[:,:-1])

train_x = np.concatenate((train[:,:-1], train_x), axis=1)
test_x = np.concatenate((test[:,:-1], test_x), axis=1)

oa, aa, k = train_test_evaluator.evaluate_train_test_pair(train_x, train[:,-1], test_x, test[:,-1])
print(oa, aa, k)
