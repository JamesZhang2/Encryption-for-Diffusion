from bfv.bfv import BFV
import numpy as np
from functools import reduce


class FHE_VEC:
    def __init__(self):
        self.bfv = BFV(n=64, param_t=20, param_q=63)
        self.sk, self.pk, self.ek = self.bfv.key_gen()
        self.unit = 200

    def encrypt(self, x: np.ndarray) -> np.ndarray:
        encrypt = np.vectorize(self._encrypt)
        return encrypt(x)

    def _encrypt(self, x: int):  # encrypt integer
        '''encrypts integer into ciphertext'''
        x_poly = self.bfv.encode_int(x, unit=self.unit)
        c1, c2 = self.bfv.encrypt(self.pk, x_poly)
        return np.array([c1, c2])

    def decrypt(self, c: np.ndarray) -> np.ndarray:
        decrypt = np.vectorize(self._decrypt)
        return decrypt(c)

    def _decrypt(self, c):
        '''decrypts ciphertext into integer'''
        x_poly = self.bfv.decrypt(self.sk, c)
        return self.bfv.decode_int(x_poly, unit=self.unit)

    def negate(self, c: np.ndarray) -> np.ndarray:
        '''negates ciphertext'''
        negate = np.vectorize(self._negate)
        return negate(c)

    def _negate(self, c):
        '''negates ciphertext'''
        return self.bfv.eval_negate(self.ek, c)

    def add(self, c1, c2):
        add = np.vectorize(self._add)
        return add(c1, c2)

    def _add(self, c1, c2):
        '''homomorphically adds two ciphertexts'''
        c1, c2 = self.bfv.eval_add(self.ek, c1, c2)
        return np.array([c1, c2])

    def add_const(self, c, p):
        '''homormophically adds ciphertext c and plaintext p'''
        add_const = np.vectorize(self._add_const)
        return add_const(c, p)

    def _add_const(self, c, p):
        poly = self.bfv.encode_int(p, unit=self.unit)
        c1, c2 = self.bfv.eval_add_const(self.pk, self.ek, poly, c)
        return np.array([c1, c2])

    def mult(self, c1, c2):
        mult = np.vectorize(self._mult)
        return mult(c1, c2)

    def _mult(self, c1, c2):
        '''homormophically multiplies two ciphertexts'''
        c_triple = self.bfv.eval_mult(self.ek, c1, c2, relin=False)
        plain = self.bfv.decrypt_raw_3(self.sk, c_triple)
        decoded_int = self.bfv.decode_int(plain, unit=self.unit ** 2)
        enc_int = self._encrypt(decoded_int)
        return enc_int

    def mult_const(self, c, p):
        mult_const = np.vectorize(self._mult_const)
        return mult_const(c, p)

    def _mult_const(self, c, p):
        '''homormophically multiplies ciphertext and integer'''
        y_poly = self.bfv.encode_int(p, unit=self.unit)
        c_triple = self.bfv.eval_mult_const(self.pk, self.ek, y_poly, c)
        plain = self.bfv.decrypt_raw_3(self.sk, c_triple)
        decoded_int = self.bfv.decode_int(plain, unit=self.unit ** 2)
        enc_int = self._encrypt(decoded_int)
        return enc_int

    def pow(self, c, power):
        pow = np.vectorize(self._pow)
        return pow(c, power)

    def _pow(self, c, power):  # positive integer
        '''homormophically raises ciphertext to positive integer >= 1'''
        assert power >= 1
        if power == 1:
            return c
        return self.mult(c, pow(c, power=power-1))

    def sum(self, c: np.ndarray):
        return reduce(self._add, c)

    def dot(self, c1: np.ndarray, c2: np.ndarray):
        assert len(c1) == len(c2)
        mults = self.mult(c1, c2)
        sum = self.sum(mults)
        return sum

    def dot_const(self, c: np.ndarray, p: np.ndarray):
        mults = self.mult_const(c, p)
        sum = self.sum(mults)
        return sum

    def matrix_product(self, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        """
        Homomorphically computes the matrix product of two ciphertext matrices.
        c1 shape: (m, n, 2)
        c2 shape: (n, p, 2)
        Output shape: (m, p, 2)
        """
        assert c1.shape[1] == c2.shape[0], f"Dim mismatch of mat shape {c1.shape} and {c2.shape}"

        m, n = c1.shape[0], c2.shape[1]

        result = np.empty((m, n), dtype=object)

        for i in range(m):
            for j in range(n):
                row = c1[i]
                col = c2[:, j]
                result[i, j] = self.dot(row, col)

        return result

    def matrix_product_const(self, c: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Homomorphically computes the matrix product of ciphertext and plaintext matrices.
        ciphertext c shape: (m, p)
        plaintext p shape: (p, n)
        Output shape: (m, n)
        """
        assert c.shape[1] == p.shape[0], f"Dim mismatch of mat shape {c.shape} and {p.shape}"

        m, n = c.shape[0], p.shape[1]

        result = np.empty((m, n), dtype=object)

        for i in range(m):
            for j in range(n):
                row = c[i]
                col = p[:, j]
                result[i, j] = self.dot_const(row, col)

        return result
