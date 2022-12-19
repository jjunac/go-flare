#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd

confmat = torchmetrics.ConfusionMatrix(task="binary", num_classes=2)

class OilSpillDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv(".datasets/oil_spill.csv").values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        targets = [0, 0]
        targets[1 if self.data[i][49] > 0.5 else 0] = 1
        return self.data[i][:49], torch.FloatTensor(targets)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(49, 25),
            nn.ReLU(),
            nn.Linear(25, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = x.to(torch.float32)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss


def test_loop(test_data, model, loss_fn, print_progress):
    dataloader = DataLoader(test_data, batch_size=len(test_data))
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            if print_progress:
                print(confmat(pred.argmax(1), y.argmax(1)))

    test_loss /= num_batches
    correct /= size
    if print_progress:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


model = NeuralNetwork()

learning_rate = 1e-3
batch_size = 100
epochs = 1000000000

data = OilSpillDataset()
training_data, test_data = random_split(data, [0.7, 0.3])

train_dataloader = DataLoader(training_data, batch_size=len(training_data))
# test_dataloader = DataLoader(test_data, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    if t%100 == 0:
        print(f"Epoch {t+1}\n-------------------------------")
    training_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_data, model, loss_fn, print_progress=(t%100 == 0))
print("Done!")