import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import train_test_evaluator

df = pd.read_csv('indian_pines.csv')
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
data = df.to_numpy()
train, test = train_test_split(data, test_size=0.2, random_state=1)
oa, aa, k = train_test_evaluator.evaluate_train_test_pair(train[:,:-1], train[:,-1], test[:,:-1], test[:,-1])
print(oa, aa, k)


diff = np.diff(data[:,0:-1], axis=1)
adiff = np.abs(diff)
adiff_mean = np.mean(adiff, axis=0)
adiff_padded = np.concatenate(([adiff_mean[0]] , adiff_mean , [adiff_mean[-1]]))

adiff_mean = (adiff_padded[:-1] + adiff_padded[1:]) / 2
s_weights = np.argsort(adiff_mean)
filtered_indices = s_weights[:30]

oa, aa, k = train_test_evaluator.evaluate_train_test_pair(train[:,filtered_indices], train[:,-1], test[:,filtered_indices], test[:,-1])
print(oa, aa, k)

random_numbers = np.random.choice(np.arange(200), size=30, replace=False)
oa, aa, k = train_test_evaluator.evaluate_train_test_pair(train[:,random_numbers], train[:,-1], test[:,random_numbers], test[:,-1])
print(oa, aa, k)

# diff = np.gradient(data[:,0:-1], axis=1)
# diff = np.gradient(diff, axis=1)
# adiff = np.abs(diff)
# weights = np.mean(adiff, axis=0)
# s_weights = np.argsort(weights)
# filtered_indices = s_weights[-30:]
# oa, aa, k = train_test_evaluator.evaluate_train_test_pair(train[:,filtered_indices], train[:,-1], test[:,filtered_indices], test[:,-1])
# print(oa, aa, k)
