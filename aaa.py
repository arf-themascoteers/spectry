import spectral
import numpy as np
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering, KMeans
import train_test_evaluator

df = pd.read_csv('indian_pines.csv')
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
X = df.iloc[:, :-1].to_numpy()

t = 30
clustering = SpectralClustering(n_clusters=t, affinity='nearest_neighbors', assign_labels='kmeans')
clusters = clustering.fit_predict(X.T)
data = df.to_numpy()
train, test = train_test_split(data, test_size=0.2, random_state=1)
oa, aa, k = train_test_evaluator.evaluate_train_test_pair(train[:,:-1], train[:,-1], test[:,:-1], test[:,-1])
print(oa, aa, k)

for i in range(10):
    closest_elements = []
    for i in range(t):
        cluster_indices = np.where(clusters == i)[0]
        cluster_elements = X.T[cluster_indices]
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(cluster_elements)
        centroid = kmeans.cluster_centers_[0]
        closest_idx = np.argmin(np.linalg.norm(cluster_elements - centroid, axis=1))
        closest_elements.append(cluster_indices[closest_idx])

    print(closest_elements)


    oa, aa, k = train_test_evaluator.evaluate_train_test_pair(train[:,closest_elements], train[:,-1], test[:,closest_elements], test[:,-1])
    print(oa, aa, k)


    random_numbers = np.random.choice(np.arange(200), size=t, replace=False)
    oa, aa, k = train_test_evaluator.evaluate_train_test_pair(train[:,random_numbers], train[:,-1], test[:,random_numbers], test[:,-1])
    print(oa, aa, k)
