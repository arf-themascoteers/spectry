import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def cosine_similarity(X):
    norm = np.linalg.norm(X, axis=0)
    X_norm = X / norm
    similarity_matrix = np.dot(X_norm.T, X_norm)
    return similarity_matrix


if __name__ == "__main__":
    df = pd.read_csv('indian_pines.csv')
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    X = df.iloc[:, :-1].to_numpy()

    similarity_matrix = cosine_similarity(X)

    df_similarity = pd.DataFrame(similarity_matrix)
    df_similarity.to_csv('similarity_matrix6.csv', index=False, header=False)
