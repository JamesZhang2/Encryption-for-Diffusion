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
from models import FHELinear
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from fhe import FHE
from models import scale_round
from sklearn.datasets import load_iris

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(42)
random.seed(0)


def download_MNIST():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True,
                              download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False,
                             download=True, transform=transform)
    return train_ds, test_ds


def download_iris():
    # Load Iris dataset
    iris = load_iris()
    data = iris.data
    targets = iris.target

    # Split into train and test datasets
    train_data, test_data, train_targets, test_targets = train_test_split(
        data, targets, test_size=0.2, random_state=42)

    # Convert to PyTorch datasets
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.long)
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.tensor(test_data, dtype=torch.float32),
        torch.tensor(test_targets, dtype=torch.long)
    )

    return train_ds, test_ds


def train(model, full_train_ds, epochs=5, learning_rate=0.001):
    # Get train and validation splits
    train_indices, val_indices = train_test_split(
        list(range(len(full_train_ds))), test_size=0.2, random_state=0)

    train_ds = torch.utils.data.Subset(full_train_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_train_ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=64,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64,
                            shuffle=False, num_workers=0)

    # Use Linear layer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
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
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = model(data)
                output = F.softmax(output, dim=-1)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total
        print(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


def test(model, full_test_ds, batch_size=64):

    test_loader = DataLoader(full_test_ds, batch_size=batch_size,
                             shuffle=True, num_workers=0)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            output = F.softmax(output, dim=-1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}, {correct}/{total}")


def test_fhe(model, fhe: FHE, full_test_ds, batch_size=4, scale=100.0):
    test_loader = DataLoader(full_test_ds, batch_size=batch_size,
                             shuffle=True, num_workers=0)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            data = data.numpy()

            print("Got Batch!")
            scale_data = scale_round(data, scale)
            enc_data = fhe.encrypt_mat(scale_data)

            print("Forwarding through model!")
            enc_output, out_scale = model(enc_data, scale)

            print("Decrypting output!")
            output = fhe.decrypt_mat(enc_output)
            # output = F.softmax(output, dim=-1)
            pred = np.argmax(output, axis=1).reshape(-1, 1)
            # print("Predictions:", pred)
            # print("Target:", target.cpu().numpy())
            correct += np.sum(pred == target.cpu().numpy().reshape(-1, 1))
            total += target.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}, {correct}/{total}")


if __name__ == "__main__":
    # model = Linear(28 * 28, 10).to(device)
    # fhe = FHE()
    # fhe_model = FHELinear(model, fhe)
    # train_ds, test_ds = download_MNIST()
    # train(model, train_ds, epochs=1) # 1 for debugging purposes
    # test(model, test_ds)

    model = Linear(4, 3).to(device)
    train_ds, test_ds = download_iris()
    # 1 for debugging purposes
    train(model, train_ds, epochs=100, learning_rate=1e-2)
    test(model, test_ds, batch_size=64)
    fhe = FHE()
    fhe_model = FHELinear(model, fhe)
    # print("linear model weights", model.linear.weight, model.linear.bias)
    # print("fhe model weights", fhe_model.weight, fhe_model.true_bias)
    test_fhe(fhe_model, fhe, test_ds, batch_size=64, scale=100)
    test_fhe(fhe_model, fhe, test_ds, batch_size=64, scale=10000)
    test_fhe(fhe_model, fhe, test_ds, batch_size=64, scale=10000000)
