import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import simple_ann
import complex_ann


df = pd.read_csv('indian_pines.csv')
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
data = df.to_numpy()
train, test = train_test_split(data, test_size=0.2, random_state=1)
#oa, aa, k = simple_ann.SimpleANN.fit(train[:,:-1], train[:,-1], test[:,:-1], test[:,-1])
oa, aa, k = complex_ann.ComplexANN.fit(train[:,:-1], train[:,-1], test[:,:-1], test[:,-1])
print(oa, aa, k)
