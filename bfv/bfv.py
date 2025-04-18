from sage.all import *
import numpy as np
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler
import sympy
import random

class BFV():
    def __init__(self, n, param_t=None, param_q=None, t=None, q=None):
        '''
        if param_t is specified, randomly generate a prime for t with that number of bits.
        Similar for param_q.
        t is the plaintext coefficient; must be prime
        q is the ciphertext coefficient; must be prime
        n is the degree (ring dimension); usually a power of 2
        Usually q >> t
        plaintext ring: P = Z_t[x]/(x^n + 1)
        ciphertext ring: C = R_q X R_q where R_q = Z_q[x]/(x^n + 1)
        '''
        if param_t != None:
            t = sympy.randprime(2 ** param_t, 2 ** (param_t + 1))
        if param_q != None:
            q = sympy.randprime(2 ** param_q, 2 ** (param_q + 1))
        self.t = t
        self.q = q
        self.n = n
        Ztx = PolynomialRing(GF(t), 'x')
        x = Ztx.gen()
        self.P_ring = Ztx.quotient(x ** n + 1)
        Zqx = PolynomialRing(GF(q), 'x')
        x = Zqx.gen()
        self.R_q = Zqx.quotient(x ** n + 1)
        self.C_ring = (self.R_q, self.R_q)

    ##### Utility functions #####

    def array_to_P_ring(self, arr):
        '''
        Converts a np.array to a P_ring element, where arr[i] is the coefficient of x^i
        '''
        if len(arr) > self.n:
            raise ValueError("Input array length larger than dimension")
        xbar = self.P_ring.gen()
        return sum([(arr[i] % self.t) * xbar ** i for i in range(len(arr))])
    
    def list_to_P_ring(self, lst):
        '''
        Converts a Python list to a P_ring element, where lst[i] is the coefficient of x^i
        '''
        if len(lst) > self.n:
            raise ValueError("Input list length larger than dimension")
        xbar = self.P_ring.gen()
        return sum([(lst[i] % self.t) * xbar ** i for i in range(len(lst))])

    def poly_to_array(self, poly):
        '''
        Converts a polynomial to an array, with the ith element being the coefficient of x^i
        '''
        return np.array(list(poly), dtype=int)

    def diff_P_ring(self, p1, p2) -> np.array:
        '''
        Requires: p1 and p2 are polynomials of the P_ring
        Returns an array whose ith entry is the absolute difference
        between the x^i coefficient of p1 and p2, where we consider the wrap-around
        and take the smaller of the two differences.
        '''
        arr1 = self.poly_to_array(p1)
        arr2 = self.poly_to_array(p2)
        return np.min(np.array([abs(arr1 - arr2), abs(arr1 + self.t - arr2), abs(arr1 - self.t - arr2)]), axis=0)

    ##### Functions to sample from distributions #####

    def _sample_error_dtbn(self, mu=0, sigma=8/sqrt(2 * pi), beta=19):
        '''
        Draws a random sample from the error distribution (discrete Gaussian) with the given parameters
        '''
        sampler = DiscreteGaussianDistributionIntegerSampler(sigma=sigma, c=mu, tau=beta)
        return sampler()
    
    def _sample_from_R2(self):
        '''
        Draws a random sample from R_2, a polynomial of degree n with coefficients in [-1, 0, 1]
        '''
        return np.array([np.random.choice([-1, 0, 1]) for _ in range(self.n)])

    ##### Functions for encryption #####

    def key_gen(self): # -> tuple[np.array, C_ring, C_ring]
        '''
        Returns (sk, pk, ek)
        sk (secret key) is in R_2
        pk (public key) = (pk1, pk2) is in (R_q, R_q)
        ek (evaluation key) = (ek1, ek2) is in (R_q, R_q)
        '''
        sk = np.array([np.random.choice([-1, 0, 1]) for _ in range(self.n)])
        xbar = self.R_q.gen()
        a = sum([random.randint(0, self.q) * (xbar ** i) for i in range(self.n)])
        sk_q = self.R_q(list(sk))  # sk cast to R_q
        pk1 = -(a*sk_q + self._sample_error_dtbn())
        pk2 = a
        pk = (pk1, pk2)
        ek = (pk1 + sk_q ** 2, pk2)
        return (sk, pk, ek)

    def encrypt(self, pk, m): # -> C_ring
        '''
        Encrypts m (message) into c (its ciphertext) using pk (the public key)
        '''
        u = self._sample_from_R2()
        u_q = self.R_q(list(u))  # cast to R_q
        # print("u_q:", u_q)
        # print("pk:", pk)
        # print("m:", m)
        # print("t:", self.t)
        # print("q:", self.q)
        pk1, pk2 = pk
        delta = self.q // self.t
        # print("delta:", delta)
        c1 = pk1 * u_q + self._sample_error_dtbn() + delta * self.R_q(m.list())
        c2 = pk2 * u_q + self._sample_error_dtbn()
        # print("c1, c2:", c1, c2)
        return (c1, c2)

    def decrypt(self, sk, c): # -> P_ring
        '''
        Decrypts c (ciphertext) into m (its message) using sk (the secret key)
        '''
        c1, c2 = c
        sk_q = self.R_q(list(sk))  # sk cast to R_q
        # print("sk_q:", sk_q)
        # print("arr:", np.array((c1 + c2 * sk_q).list(), dtype=int))
        # print("t:", self.t)
        # print("q:", self.q)
        # print("arr * t / q:", np.array((c1 + c2 * sk_q).list(), dtype=int) * self.t / self.q)
        lst = (c1 + c2 * sk_q).list()
        # print("lst:", lst)
        # print("times:", lst[0] * self.t)
        # print("q:", self.q)
        # print("times div:", int(int(lst[0]) * int(self.t) / int(self.q)))
        coeffs = [int(int(num) * int(self.t) / int(self.q)) for num in (c1 + c2 * sk_q).list()]
        # print(coeffs)
        return self.list_to_P_ring(coeffs)
    
    def eval_add(self, ek, c1, c2): # -> C_ring
        '''
        If c1 is an encryption of m1 and c2 is an encryption of m2,
        outputs a ciphertext c3 encrypting (m1 + m2)
        '''
        return (c1[0]+c2[0], c1[1]+c2[1])
    
    def eval_mult(self, ek, c1, c2): # -> tuple[R_q, R_q]
        '''
        If c1 is an encryption of m1 and c2 an encryption of m2,
        outputs a ciphertext encrypting (m1 * m2)
        '''
        a = self.t*(c1[0]*c2[0])/self.q
        b = self.t*(c1[0]*c2[1] + c1[1]*c2[0])/self.q
        c = self.t*(c1[1]*c2[1])/self.q
        return self._relinearize(ek, (a, b, c))

    # TODO: Implement eval_add_const and eval_mult_const - should be trivial

    def _relinearize(self, ek, c): # C_ring:
        return (c[0]+ek*c[2], c[1]+ek*c[2])

# n = 4  # degree
# t = 31  # plaintext coefficient
# # q = 1217  # ciphertext coefficient
# q = 139

# bfv = BFV(n=8, param_t=6, param_q=10)
# (sk, pk, ek) = bfv.key_gen()
# print("sk:", sk)
# print("pk:", pk)
# print("ek:", ek)
# m = bfv.array_to_P_ring(np.array([12, 27, 2, 17]))
# print("m:", m)
# enc = bfv.encrypt(pk, m)
# print("Encrypted m:", enc)
# dec = bfv.decrypt(sk, enc)
# print("Decrypted m:", dec)
