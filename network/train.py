# train_cfg.py

import os
import random
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Linear
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def download_data():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True,
                              download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False,
                             download=True, transform=transform)
    return train_ds, test_ds


def train():
    # Get train and validation splits
    full_train_ds, test_ds = download_data()

    train_indices, val_indices = train_test_split(
        list(range(len(full_train_ds))), test_size=0.2, random_state=0)

    train_ds = torch.utils.data.Subset(full_train_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_train_ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=64,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64,
                            shuffle=False, num_workers=0)

    # Use Linear layer
    model = Linear(28 * 28, 10).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten the input

            optimizer.zero_grad()
            output = model(data)
            output = F.softmax(output, dim=1)  # get log prob.
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss / len(train_loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = model(data)
                output = F.softmax(output, dim=-1)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")


if __name__ == "__main__":
    train()
