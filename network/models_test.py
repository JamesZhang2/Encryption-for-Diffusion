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
from fhe_vec import VEC_FHE

fhe = FHE()
vec_fhe = VEC_FHE()


def test_encrypt_then_decrypt():
    # Encrypt and decrypt a single integer
    print("Running encrypt_then_decrypt...")
    max_error = 0
    for _ in range(100):
        n = 100
        c = fhe.encrypt(n)
        dec_n = fhe.decrypt(c)
        max_error = max(max_error, abs(n - dec_n))
    print("max_error:", max_error)
    # Encrypt and decrypt a numpy array

    numpy_array = np.random.randint(-100, 100, size=(2, 5))
    # Encrypt and decrypt the numpy array
    encrypted_array = vec_fhe.encrypt(numpy_array)
    decrypted_array = vec_fhe.decrypt(encrypted_array)

    assert np.allclose(
        numpy_array, decrypted_array), "Decrypted array does not match the original array"


def test_multiplication():
    numpy_array1 = np.random.randint(-100, 100, size=(2, 5))
    numpy_array2 = np.random.randint(-100, 100, size=(5, 3))

    product = np.dot(numpy_array1, numpy_array2)
    print(product)

    a = vec_fhe.encrypt(numpy_array1)
    b = vec_fhe.encrypt(numpy_array2)
    c = vec_fhe.matrix_product(a, b)
    decrypted_product = vec_fhe.decrypt(c)

    print(type(decrypted_product))
    assert np.allclose(
        product, decrypted_product), "Decrypted product does not match the original product"


def test_addition():
    numpy_array1 = np.random.randint(-100, 100, size=(2, 5))
    numpy_array2 = np.random.randint(-100, 100, size=(2, 5))

    sum = numpy_array1 + numpy_array2
    print(sum)

    # a = vec_fhe.encrypt(numpy_array1)
    # b = vec_fhe.encrypt(numpy_array2)
    # c = vec_fhe.mult(a, b)
    # decrypted_product = fhe.decrypt(c)
    a = fhe.encrypt_mat(numpy_array1)
    b = fhe.encrypt_mat(numpy_array2)
    c = fhe.add_mat(a, b)
    decrypted_sum = fhe.decrypt_mat(c)

    print(decrypted_sum)
    assert np.allclose(
        sum, decrypted_sum), "Decrypted product does not match the original product"


if __name__ == "__main__":
    test_encrypt_then_decrypt()
    test_multiplication()
    # test_addition()
