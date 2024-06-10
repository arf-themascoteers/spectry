import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import train_test_evaluator


class Att2(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.multihead_attn = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(200, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16)
        )
        self.layer_norm = nn.LayerNorm(200)

    def forward(self, X):
        X = X.unsqueeze(2)
        attn_output, attentions = self.multihead_attn(X, X, X)
        attn_output = torch.mean(attn_output, dim=2)
        attn_output = self.layer_norm(attn_output)
        x = self.fc(attn_output)
        similarity_matrix = attentions.mean(dim=0)
        return x, similarity_matrix

    def fit(self, train, test):
        X_train = torch.tensor(train[:,0:-1], dtype=torch.float32)
        y_train = torch.tensor(train[:,-1], dtype=torch.int32).type(torch.LongTensor)
        X_test = torch.tensor(test[:,0:-1], dtype=torch.float32)
        y_test = torch.tensor(test[:,-1], dtype=torch.int32).type(torch.LongTensor)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in range(100):
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

        with torch.no_grad():
            y_hat, sm = model(X_test)
            y_hat = torch.argmax(y_hat, dim=1)
            y_hat = y_hat.cpu().detach().numpy()
            y = y_test.cpu().detach().numpy()
            return train_test_evaluator.calculate_metrics(y, y_hat)

if __name__ == "__main__":
    df = pd.read_csv('indian_pines.csv')
    df.iloc[:, -1], class_labels = pd.factorize(df.iloc[:, -1])
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    data = df.to_numpy()
    train, test = train_test_split(data, test_size=0.2, random_state=1)
    model = Att2(200)
    o,a,k = model.fit(train, test)
    print(o,a,k)
