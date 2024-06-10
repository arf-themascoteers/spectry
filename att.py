import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Att(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.multihead_attn = nn.MultiheadAttention(embed_dim=8, num_heads=8, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(True),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 200)
        )
        self.layer_norm = nn.LayerNorm(200)  # Using LayerNorm only

    def forward(self, X):
        X = X.unsqueeze(2)
        X = X.repeat(1, 1, 8)
        attn_output, attentions = self.multihead_attn(X, X, X)
        attn_output = torch.mean(attn_output, dim=2)
        attn_output = self.layer_norm(attn_output)  # Normalizing
        x = self.encoder(attn_output)
        x = self.decoder(x)
        similarity_matrix = attentions.mean(dim=0)
        return x, similarity_matrix

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        dataset = TensorDataset(X, X)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in range(5):
            for batch_idx, (batch_X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                output, sm = self(batch_X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')
            sm = sm.detach().numpy()
            df = pd.DataFrame(sm)
            df.to_csv(f'output_{epoch}.csv', index=False, header=False)


if __name__ == "__main__":
    df = pd.read_csv('indian_pines.csv')
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    X = df.iloc[:, :-1].to_numpy()
    model = Att(X.shape[1])
    model.fit(X)
