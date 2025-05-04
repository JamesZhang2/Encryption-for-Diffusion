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
from fhe_vec import FHE_VEC

fhe = FHE()
fhe_vec = FHE_VEC()


def test_encrypt_then_decrypt():
    # Encrypt and decrypt a single integer
    print("Running encrypt_then_decrypt test...")
    max_error = 0
    for _ in range(100):
        n = 100
        c = fhe_vec.encrypt(np.array([n]))
        dec_n = fhe_vec.decrypt(c)
        max_error = max(max_error, abs(n - dec_n))
    print("max_error:", max_error)
    # Encrypt and decrypt a numpy array

    numpy_array = np.random.randint(-100, 100, size=(2, 5))
    # Encrypt and decrypt the numpy array
    encrypted_array = fhe_vec.encrypt(numpy_array)
    decrypted_array = fhe_vec.decrypt(encrypted_array)

    assert np.allclose(
        numpy_array, decrypted_array), "Decrypted array does not match the original array"

    print("Encrypt_then_decrypt test passed")


def test_matmul():
    print("Running matmul test...")
    numpy_array1 = np.random.randint(-100, 100, size=(2, 5))
    numpy_array2 = np.random.randint(-100, 100, size=(5, 3))

    product = np.dot(numpy_array1, numpy_array2)

    a = fhe_vec.encrypt(numpy_array1)
    b = fhe_vec.encrypt(numpy_array2)
    c = fhe_vec.matrix_product(a, b)
    decrypted_product = fhe_vec.decrypt(c)

    assert np.allclose(
        product, decrypted_product), "Decrypted product does not match the original product"

    print("Matmul test passed")


def test_matmul_const():
    print("Running matmul const test...")
    numpy_array1 = np.random.randint(-100, 100, size=(2, 5))
    numpy_array2 = np.random.randint(-100, 100, size=(5, 3))

    product = np.dot(numpy_array1, numpy_array2)

    a = fhe_vec.encrypt(numpy_array1)
    b = numpy_array2
    c = fhe_vec.matrix_product_const(a, b)
    decrypted_product = fhe_vec.decrypt(c)

    assert np.allclose(
        product, decrypted_product), "Decrypted product does not match the original product"

    print("Matmul const test passed")


def test_add():
    print("Running add test...")
    numpy_array1 = np.random.randint(-100, 100, size=(2, 5))
    numpy_array2 = np.random.randint(-100, 100, size=(2, 5))

    sum = numpy_array1 + numpy_array2
    print(sum)

    a = fhe_vec.encrypt(numpy_array1)
    b = fhe_vec.encrypt(numpy_array2)
    c = fhe_vec.add(a, b)
    decrypted_sum = fhe_vec.decrypt(c)

    print(decrypted_sum)
    assert np.allclose(
        sum, decrypted_sum), "Decrypted product does not match the original product"
    print("Add test passed")


def test_add_const():
    print("Running add constant test...")
    numpy_array1 = np.random.randint(-100, 100, size=(2, 5))
    numpy_array2 = np.random.randint(-100, 100, size=(2, 5))

    sum = numpy_array1 + numpy_array2
    print(sum)

    a = fhe_vec.encrypt(numpy_array1)
    b = numpy_array2
    c = fhe_vec.add_const(a, b)
    decrypted_sum = fhe_vec.decrypt(c)

    print(decrypted_sum)
    assert np.allclose(
        sum, decrypted_sum), "Decrypted product does not match the original product"
    print("Add constant test passed")


if __name__ == "__main__":
    test_encrypt_then_decrypt()
    test_matmul()
    test_matmul_const()
    test_add()
    test_add_const()
