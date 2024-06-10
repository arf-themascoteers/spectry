import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import train_test_evaluator
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class ComplexANN(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.multihead_attn = nn.MultiheadAttention(embed_dim=200, num_heads=8)

    def forward(self, X):
        X = X.transpose(0, 1)
        attn_output, attn_output_weights = self.multihead_attn(X, X, X)
        similarity_matrix = attn_output_weights.mean(dim=0)
        return similarity_matrix


if __name__ == "__main__":
    df = pd.read_csv('indian_pines_min.csv')
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    data = df.to_numpy()
    train, test = train_test_split(data, test_size=0.2, random_state=1)
    model = ComplexANN(200)
    sim = model(train[:,:-1])
    print(sim.shape)
