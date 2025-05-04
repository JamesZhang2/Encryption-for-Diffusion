from bfv.bfv import BFV
import numpy as np


class VEC_FHE:
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

    def decrypt(self, x: np.ndarray) -> np.ndarray:
        decrypt = np.vectorize(self._decrypt)
        return decrypt(x)

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
        return self.bfv.eval_add(self.ek, c1, c2)

    def add_const(self, c, p):
        add_const = np.vectorize(self._add_const)
        return add_const(c, p)

    def _add_const(self, c, p):
        '''homormophically adds ciphertext and integer'''
        poly = self.bfv.encode_int(p, unit=self.unit)
        return self.bfv.eval_add_const(self.pk, self.ek, poly, c)

    def mult(self, c1, c2):
        mult = np.vectorize(self._mult)
        return mult(c1, c2)

    def _mult(self, c1, c2):
        '''homormophically multiplies two ciphertexts'''
        c_triple = self.bfv.eval_mult(self.ek, c1, c2, relin=False)
        plain = self.bfv.decrypt_raw_3(self.sk, c_triple)
        decoded_int = self.bfv.decode_int(plain, unit=self.unit ** 2)
        encoded_poly = self.bfv.encode_int(decoded_int, unit=self.unit)
        return self.bfv.encrypt(self.pk, encoded_poly)

    def mult_const(self, c, y):
        mult_const = np.vectorize(self._mult_const)
        return mult_const(c, y)

    def _mult_const(self, c, y):
        '''homormophically multiplies ciphertext and integer'''
        y_poly = self.bfv.encode_int(y, unit=self.unit)
        c_triple = self.bfv.eval_mult_const(self.pk, self.ek, y_poly, c)
        plain = self.bfv.decrypt_raw_3(self.sk, c_triple)
        decoded_int = self.bfv.decode_int(plain, unit=self.unit ** 2)
        encoded_poly = self.bfv.encode_int(decoded_int, unit=self.unit)
        return self.bfv.encrypt(self.pk, encoded_poly)

    def pow(self, c, power):
        pow = np.vectorize(self._pow)
        return pow(c, power)

    def _pow(self, c, power):  # positive integer
        '''homormophically raises ciphertext to positive integer >= 1'''
        assert power >= 1
        if power == 1:
            return c
        return self.mult(c, pow(c, power=power-1))

    def matrix_product(self, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        '''homomorphically multiplies two ciphertexts'''
        assert c1.shape[1] == c2.shape[0]
        result = np.zeros((c1.shape[0], c2.shape[1]), dtype=object)
        for i in range(c1.shape[0]):
            for j in range(c2.shape[1]):
                result[i, j] = self.mult(c1[i], c2[:, j])
        return result
