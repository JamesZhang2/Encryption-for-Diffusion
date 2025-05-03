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

    def center_mod_q(self, x):
        q = self.q
        x = int(x) % q
        return x if x < q // 2 else x - q

    # def round_half_up(self, i):
    #     return int(i + 0.5) if i >= 0 else int(i - 0.5)

    def array_to_P_ring(self, arr):
        '''
        Converts a np.array to a P_ring element, where arr[i] is the coefficient of x^i
        '''
        if len(arr) > self.n:
            raise ValueError("Input array length larger than dimension:", arr)
        if len(arr) == 0:
            raise ValueError("Input array is empty")
        xbar = self.P_ring.gen()
        return sum([(arr[i] % self.t) * xbar ** i for i in range(len(arr))])
    
    def list_to_P_ring(self, lst):
        '''
        Converts a Python list to a P_ring element, where lst[i] is the coefficient of x^i
        '''
        if len(lst) > self.n:
            raise ValueError("Input list length larger than dimension:", lst)
        if len(lst) == 0:
            raise ValueError("Input list is empty")
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

    def encode_int(self, num, unit=None):
        '''
        Encodes an integer using binary,
        with unit being the difference between the encoding of 0 and the encoding of 1.
        Returns a P_ring element (plaintext).
        For example, if unit is 100, then 3 is encoded as 100 + 100x,
        14 is encoded as 100x + 100x^3,
        and -5 is encoded as -100 + -100x^2.
        Note that we're wrapping around, that is,
        if the coefficient is greater than self.t/2, we treat it as negative.
        Requires: unit <= self.t/3 (because we need to at least distinguish -1, 0, and 1)
        It is recommended that unit << self.t/3 so that we don't wrap around.
        If unit is unspecified, defaults to min(1000, self.t // 10)
        Requires: abs(num) < 2 ** self.n so that we can represent the number in binary.
        '''
        if unit == None:
            unit = min(1000, self.t // 10)
        if num == 0:
            return self.list_to_P_ring([0])
        neg_sign = 1 if num > 0 else -1
        num = abs(num)
        lst = []
        while num != 0:
            lst.append((num % 2) * neg_sign * unit)
            num //= 2
        return self.list_to_P_ring(lst)
    
    def decode_int(self, poly, unit=None) -> int:
        '''
        Decodes a P_ring element to an integer,
        with unit being the difference between the encoding of 0 and the encoding of 1.
        Returns an integer.
        The same unit should be used for encoding and decoding.
        Note that we're wrapping around, that is,
        if the coefficient is greater than self.t/2, we treat it as negative.
        Requires: unit <= self.t/3 (because we need to at least distinguish -1, 0, and 1)
        It is recommended that unit << self.t/3 so that we don't wrap around.
        If unit is unspecified, defaults to min(1000, self.t // 10)
        '''
        if unit == None:
            unit = min(1000, self.t // 10)
        lst = list(poly)
        # print(lst)
        num = 0
        for i in range(len(lst)):
            if int(lst[i]) <= self.t / 2:
                # treat coefficient as positive
                # round to nearest unit
                num += round(int(lst[i]) / unit) * (2 ** i)
            else:
                # treat coefficient as negative
                num += round((int(lst[i]) - self.t) / unit) * (2 ** i)
        return int(num)

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
        delta = self.q // self.t
        s2 = sk_q * sk_q
        a = self.R_q.random_element()
        e1 = self._sample_error_dtbn()
        e2 = self._sample_error_dtbn()
        ek1 = -(a * sk_q + e1) + delta * s2
        ek2 = a + e2
        ek = (ek1, ek2)
        return (sk, pk, ek)

    def encrypt(self, pk, m, add_noise=True): # -> C_ring
        '''
        Encrypts m (message) into c (its ciphertext) using pk (the public key)
        We don't need to add noise if we're encrypting a constant for eval_add_const and eval_mult_const
        '''
        u = self._sample_from_R2()
        u_q = self.R_q(list(u))  # cast to R_q
        pk1, pk2 = pk
        delta = self.q // self.t
        if not add_noise:
            c1 = pk1 * u_q + delta * self.R_q(m.list())
            c2 = pk2 * u_q
        else: 
            c1 = pk1 * u_q + self._sample_error_dtbn() + delta * self.R_q(m.list())
            c2 = pk2 * u_q + self._sample_error_dtbn()
        return (c1, c2)

    def decrypt(self, sk, c): # -> P_ring
        '''
        Decrypts c (ciphertext) into m (its message) using sk (the secret key)
        '''
        c1, c2 = c
        sk_q = self.R_q(list(sk))  # sk cast to R_q
        coeffs = [int(round(int(num) * self.t / self.q)) % self.t for num in (c1 + c2 * sk_q).list()]
        return self.list_to_P_ring(coeffs)
    
    def eval_negate(self, ek, c):
        c_1, c_2 = c
        return (-c_1, -c_2)
    
    def eval_add(self, ek, c1, c2): # -> C_ring
        '''
        If c1 is an encryption of m1 and c2 is an encryption of m2,
        outputs a ciphertext c3 encrypting (m1 + m2)
        '''
        return (c1[0]+c2[0], c1[1]+c2[1])

    def _mult_without_mod_q(self, p1, p2):
        '''
        p1 and p2 are elements in Z_q[x]/(x^n + 1)
        We want to multiply them as if they're in Z[x]/(x^n + 1)
        Returns an element of Z[x]/(x^n + 1)
        '''
        R = PolynomialRing(ZZ, 'y')
        y = R.gen()
        f = y ** self.n + 1
        R = R.quotient(f, 'ybar')
        coeff1 = list(p1)
        coeff2 = list(p2)
        py1 = R(coeff1)
        py2 = R(coeff2)
        return py1 * py2
    
    def eval_mult(self, ek, c1, c2, relin=False): # -> tuple[R_q, R_q]
        '''
        If c1 is an encryption of m1 and c2 an encryption of m2,
        outputs a ciphertext encrypting (m1 * m2)
        '''
        # print("In eval_mult...")
        # print("c1:", c1)
        # print("c2:", c2)
        # print("t:", self.t)
        # print("q:", self.q)
        ls = []
        xbar = self.R_q.gen()
        for i in range(3):
            if i==0:
                p_R = self._mult_without_mod_q(c1[0], c2[0])
            elif i==1:
                p_R = self._mult_without_mod_q(c1[0], c2[1]) + self._mult_without_mod_q(c1[1], c2[0])
            else:
                p_R = self._mult_without_mod_q(c1[1], c2[1])
            coeffs = [int(round(int(c * self.t) / self.q)) for c in list(p_R)]
            poly_R_q = sum(c * xbar**j for j, c in enumerate(coeffs))
            ls.append(poly_R_q)
        # print("before relinearize: ")
        # print("c0: ", ls[0])
        # print("c1: ", ls[1])
        # print("c2: ", ls[2])
        if relin:
            return self._relinearize(ek, tuple(ls))
        else:
            return tuple(ls)

    def _relinearize(self, ek, c): # C_ring:
        c0, c1, c2 = c
        ek1, ek2 = ek
        sk_q = self.R_q(list(sk))
        c1_star = c0 + ek1 * c2
        c2_star = c1 + ek2 * c2
        print("LHS:", c0 + c1 * sk_q + c2 * sk_q * sk_q)
        print("RHS:", c1_star + c2_star * sk_q)
        return (c0 + ek1 * c2, c1 + ek2 * c2)

    def eval_add_const(self, pk, ek, m, c):
        '''
        Adds the constant plaintext m to the ciphertext c
        '''
        cons = self.encrypt(pk, m, add_noise=False)
        return self.eval_add(ek, c, cons)

    def eval_mult_const(self, pk, ek, m, c):
        '''
        Multiplies the constant plaintext m to the ciphertext c
        '''
        cons = self.encrypt(pk, m, add_noise=False)
        return self.eval_mult(ek, c, cons, relin=False)

    def _polynomialize(self, n):
        bits = bin(n)[2:]
        coeffs = [int(b) for b in reversed(bits)]
        return self.P_ring(coeffs)


    def decrypt_raw_3(self, sk, ct3):
        c0, c1, c2 = ct3
        sk_q = self.R_q(list(sk))
        m_poly = c0 + c1 * sk_q + c2 * sk_q ** 2

        coeffs = [int(round(int(c) * self.t / self.q)) for c in list(m_poly)]
        return self.list_to_P_ring(coeffs)
    


