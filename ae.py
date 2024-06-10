import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 200)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def similarity_loss(embeddings, margin=1.0):
    n = embeddings.size(0)
    sim_loss = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim_loss += torch.norm(embeddings[i] - embeddings[j], p=2)
    sim_loss /= (n * (n - 1) / 2)
    return sim_loss


def train_model(model, X, epochs=50, batch_size=128, lr=0.001, margin=1.0):
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
            output = model(batch_X)
            reconstruction_loss = criterion(output, batch_X)
            embeddings = model.encoder(batch_X)
            sim_loss = similarity_loss(embeddings, margin)
            loss = reconstruction_loss + sim_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}')

    return model.encoder


if __name__ == "__main__":
    df = pd.read_csv('indian_pines.csv')
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    X = df.iloc[:, :-1].to_numpy()

    model = Autoencoder()
    encoder_model = train_model(model, X, epochs=50)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    embeddings = encoder_model(X_tensor).detach().numpy()
    similarity_matrix = np.corrcoef(embeddings.T)

    df_similarity = pd.DataFrame(similarity_matrix)
    df_similarity.to_csv('similarity_matrix22.csv', index=False, header=False)
    print("done")