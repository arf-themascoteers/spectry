import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import train_test_evaluator


class SimpleANN(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.fc = nn.Sequential(
            nn.Linear(self.size, 150),
            nn.LeakyReLU(),
            nn.Linear(150, 16)
        )

    def forward(self, X):
        X = self.fc(X)
        return X

    @staticmethod
    def fit(X_train, y_train, X_test, y_test):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.int32).type(torch.LongTensor)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.int32).type(torch.LongTensor)
        model = SimpleANN(X_train.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999))
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=300000, shuffle=True)

        for epoch in range(500):
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                y_hat = model(X)
                y = y.type(torch.LongTensor)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
            print(epoch, loss.item())

        with torch.no_grad():
            y_hat = model(X_test)
            y_hat = torch.argmax(y_hat, dim=1)
            y_hat = y_hat.cpu().detach().numpy()
            y = y_test.cpu().detach().numpy()
            return train_test_evaluator.calculate_metrics(y, y_hat)



