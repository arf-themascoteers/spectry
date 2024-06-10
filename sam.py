import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def spectral_angle_mapper(X):
    n = X.shape[1]
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dot_product = np.dot(X[:, i], X[:, j])
            norm_i = np.linalg.norm(X[:, i])
            norm_j = np.linalg.norm(X[:, j])
            cos_theta = dot_product / (norm_i * norm_j)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            similarity_matrix[i, j] = np.arccos(cos_theta)
    return similarity_matrix


if __name__ == "__main__":
    df = pd.read_csv('indian_pines.csv')
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    X = df.iloc[:, :-1].to_numpy()

    similarity_matrix = spectral_angle_mapper(X)

    df_similarity = pd.DataFrame(similarity_matrix)
    df_similarity.to_csv('similarity_matrix4.csv', index=False, header=False)
