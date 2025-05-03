import torch
from bfv.bfv import BFV


class FHE:
    def __init__(self):
        self.bfv = BFV(n=64, param_t=20, param_q=63)
        self.sk, self.pk, self.ek = self.bfv.key_gen()
        self.unit = 200

    def encrypt(self, x: int):  # encrypt integer
        '''encrypts integer into ciphertext'''
        x_poly = self.bfv.encode_int(x, unit=self.unit)
        return self.bfv.encrypt(self.pk, x_poly)

    def decrypt(self, c):
        '''decrypts ciphertext into integer'''
        x_poly = self.bfv.decrypt(self.sk, c)
        return self.bfv.decode_int(x_poly, unit=self.unit)

    def negate(self, c):
        '''negates ciphertext'''
        return self.bfv.eval_negate(self.ek, c)

    def add(self, c1, c2):
        '''homomorphically adds two ciphertexts'''
        return self.bfv.eval_add(self.ek, c1, c2)

    def add_const(self, c, y):
        '''homormophically adds ciphertext and integer'''
        y_poly = self.bfv.encode_int(y, unit=self.unit)
        return self.bfv.eval_add_const(self.pk, self.ek, y_poly, c)

    def mult(self, c1, c2):
        '''homormophically multiplies two ciphertexts'''
        c_triple = self.bfv.eval_mult(self.ek, c1, c2, relin=False)
        plain = self.bfv.decrypt_raw_3(self.sk, c_triple)
        decoded_int = self.bfv.decode_int(plain, unit=self.unit ** 2)
        encoded_poly = self.bfv.encode_int(decoded_int, unit=self.unit)
        return self.bfv.encrypt(self.pk, encoded_poly)

    def mult_const(self, c, y):
        '''homormophically multiplies ciphertext and integer'''
        y_poly = self.bfv.encode_int(y, unit=self.unit)
        c_triple = self.bfv.eval_mult_const(self.pk, self.ek, y_poly, c)
        plain = self.bfv.decrypt_raw_3(self.sk, c_triple)
        decoded_int = self.bfv.decode_int(plain, unit=self.unit ** 2)
        encoded_poly = self.bfv.encode_int(decoded_int, unit=self.unit)
        return self.bfv.encrypt(self.pk, encoded_poly)

    def pow(self, c, power):  # positive integer
        '''homormophically raises ciphertext to positive integer >= 1'''
        assert power >= 1
        if power == 1:
            return c
        return self.mult(c, pow(c, power=power-1))

    # Vector operations
    def encrypt_vec(self, x):
        '''encrypts pytorch vector of integers element-wise'''
        c_x = torch.empty_like(x)
        for i in range(x.shape[0]):
            c_x[i] = self.encrypt(x[i])
        return c_x

    def decrypt_vec(self, c_x):
        '''decryption of pytorch vector of ciphertexts element-wise'''
        x = torch.empty_like(c_x)
        for i in range(c_x.shape[0]):
            x[i] = self.decrypt(c_x[i])
        return c_x

    def add_vec(self, c_x, c_y):
        '''homomorphically adds two encrypted vectors element-wise'''
        c_z = torch.empty_like(c_x)
        for i in range(c_x.shape[0]):
            c_z[i] = self.add(c_x[i], c_y[i])
        return c_z

    def mult_vec(self, c_x, c_y):
        '''homomorphically multiplies two encrypted vectors element-wise'''
        c_z = torch.empty_like(c_x)
        for i in range(c_x.shape[0]):
            c_z[i] = self.mult(c_x[i], c_y[i])
        return c_z

    def scale_vec(self, c_x, scale):  # scale vector by a constant
        '''homomorphically scales an encrypted vectors by scalar integer'''
        c_z = torch.empty_like(c_x)
        for i in range(c_x.shape[0]):
            c_z[i] = self.mult_const(c_x[i], scale)
        return c_z
    
    def dot_const(self, p, c):
        '''homomorphically evaluates dot product of a plaintext vector and an encrypted vector'''
        assert len(p) == len(c)
        mults = torch.empty_like(p)
        for i in range(p.shape[0]):
            mults[i] = self.mult_const(c[i], p[i])
        sum = mults[0]
        for i in range(1, p.shape[0]):
            sum = self.add_const(sum, mults[i])
        return sum

    def dot(self, c_x, c_y):
        '''homomorphically evaluates dot product of two encrypted vectors'''
        assert len(c_x) == len(c_y)
        mults = torch.empty_like(c_x)
        for i in range(c_x.shape[0]):
            mults[i] = self.mult(c_x[i], c_y[i])
        sum = mults[0]
        for i in range(1, c_x.shape[0]):
            sum = self.add(sum, mults[i])
        return sum

    # Matrix operations

    def encrypt_mat(self, x):
        '''encrypts pytorch 2d matrix of integers element-wise'''
        c_x = torch.empty_like(x)
        for i in range(x.shape[0]):
            c_x[i] = self.encrypt_vec(x[i])
        return c_x

    def decrypt_mat(self, c_x):
        '''decrypts pytorch 2d matrix of ciphertexts element-wise'''
        x = torch.empty_like(c_x)
        for i in range(c_x.shape[0]):
            x[i] = self.decrypt_vec(c_x[i])
        return x

    def add_mat(self, c_x, c_y):
        '''homomorphically adds two pytorch 2d matrix of ciphertexts element-wise'''
        assert c_x.shape == c_y.shape
        c_z = torch.empty_like(c_x)
        for i in range(c_x.shape[0]):
            c_z[i] = self.add_vec(c_x[i], c_y[i])
        return c_z

    def add_matrix_vector(self, c_x, c_y):  # add vector to rows
        '''homomorphically add a pytorch 2d matrix and vector of ciphertexts with vector [c_y] broadcast to rows (just like broadcasting)'''
        c_z = torch.empty_like(c_x)
        for i in range(c_x.shape[0]):
            c_z[i] = self.add_vec(c_x[i], c_y)
        return c_z

    def matrix_product(self, c_x, c_y):
        '''homomorphically take matrix product of two pytorch 2d matrix of ciphertexts'''
        assert c_x.shape[1] == c_y.shape[0]
        m = c_x.shape[0]
        n = c_y.shape[1]
        out = torch.empty((m, n))
        for i in range(m):
            for j in range(n):
                out[i][j] = self.dot(c_x[i], c_y[:, j])
        return out

    def matrix_product_const(self, x, c):
        '''matrix product of plaintext matrix with ciphertext matrix'''
        assert x.shape[1] == c.shape[0]
        m = x.shape[0]
        n = c.shape[1]
        out = torch.empty((m, n))
        for i in range(m):
            for j in range(n):
                out[i][j] = self.dot_const(x[i], c[:, j])
        return out