import torch
import random
from fhe import FHE


def encrypt_then_decrypt():
    print("Running encrypt_then_decrypt...")
    max_error = 0
    for _ in range(100):
        n = random.randint(-10000000, 10000000)
        fhe = FHE()
        max_error = max(max_error, abs(n - fhe.decrypt(fhe.encrypt(n))))
    print("max_error:", max_error)


def test_addition():
    print("Running test_addition...")
    max_error = 0
    for _ in range(100):
        a1 = random.randint(-10000000, 10000000)
        a2 = random.randint(-10000000, 10000000)
        fhe = FHE()
        enc1 = fhe.encrypt(a1)
        enc2 = fhe.encrypt(a2)
        enc = fhe.add(enc1, enc2)
        max_error = max(max_error, abs(fhe.decrypt(enc) - (a1 + a2)))
    print("max_error:", max_error)


def test_addition_const():
    print("Running test_addition_const...")
    max_error = 0
    for _ in range(100):
        a1 = random.randint(-10000000, 10000000)
        a2 = random.randint(-10000000, 10000000)
        fhe = FHE()
        enc1 = fhe.encrypt(a1)
        enc = fhe.add_const(enc1, a2)
        max_error = max(max_error, abs(fhe.decrypt(enc) - (a1 + a2)))
    print("max_error:", max_error)


def test_multiplication():
    print("Running test_multiplication...")
    max_error = 0
    for _ in range(100):
        a1 = random.randint(-100000, 100000)
        a2 = random.randint(-100000, 100000)
        fhe = FHE()
        enc1 = fhe.encrypt(a1)
        enc2 = fhe.encrypt(a2)
        enc = fhe.mult(enc1, enc2)
        # print(len(enc))
        max_error = max(max_error, abs(fhe.decrypt(enc) - (a1 * a2)))
    print("max_error:", max_error)


def test_multiplication_const():
    print("Running test_multiplication_const...")
    max_error = 0
    for _ in range(100):
        a1 = random.randint(-100000, 100000)
        a2 = random.randint(-100000, 100000)
        fhe = FHE()
        enc1 = fhe.encrypt(a1)
        enc = fhe.mult_const(enc1, a2)
        # print(len(enc))
        max_error = max(max_error, abs(fhe.decrypt(enc) - (a1 * a2)))
    print("max_error:", max_error)


def run_all_tests():
    encrypt_then_decrypt()
    test_addition()
    test_multiplication()
    test_addition_const()
    test_multiplication_const()


if __name__ == "__main__":
    run_all_tests()
