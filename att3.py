import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Att(nn.Module):
    def __init__(self):
        super(Att, self).__init__()
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
        self.layer_norm = nn.LayerNorm(200)

    def forward(self, X):
        X = X.unsqueeze(2).repeat(1, 1, 8)
        attn_output, attentions = self.multihead_attn(X, X, X)
        attn_output = torch.mean(attn_output, dim=2)
        attn_output = self.layer_norm(attn_output)
        x = self.encoder(attn_output)
        x = self.decoder(x)
        similarity_matrix = attentions.mean(dim=1)
        return x, similarity_matrix


def train_model(model, X, epochs=50, batch_size=128, lr=0.0001):
    X = torch.tensor(X, dtype=torch.float32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X, X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, _ in dataloader:
            optimizer.zero_grad()
            output, similarity_matrix = model(batch_X)
            loss = criterion(output, batch_X)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}')

    return similarity_matrix


if __name__ == "__main__":
    df = pd.read_csv('indian_pines.csv')
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    X = df.iloc[:, :-1].to_numpy()

    model = Att()
    similarity_matrix = train_model(model, X, epochs=50)

    similarity_matrix_np = similarity_matrix.detach().numpy()
    df_similarity = pd.DataFrame(similarity_matrix_np)
    df_similarity.to_csv('similarity_matrix.csv', index=False, header=False)
