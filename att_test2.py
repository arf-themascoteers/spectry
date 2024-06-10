import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

df = pd.read_csv('indian_pines_min.csv')
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
X = df.iloc[:, :-1].to_numpy()
X = torch.tensor(X, dtype=torch.float32)
X = X.unsqueeze(2)
#X = X.repeat(1, 1, 8)
print(X.shape)
multihead_attn = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
attn_output, attentions = multihead_attn(X, X, X)

similarity_matrix = attentions.mean(dim=0)
print(attentions.shape)
print(similarity_matrix.shape)
s = similarity_matrix.detach().numpy()
print(s)