import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    df = pd.read_csv('indian_pines.csv')
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    X = df.iloc[:, :-1].to_numpy()

    similarity_matrix = np.corrcoef(X.T)

    df_similarity = pd.DataFrame(similarity_matrix)
    df_similarity.to_csv('similarity_matrix3.csv', index=False, header=False)
