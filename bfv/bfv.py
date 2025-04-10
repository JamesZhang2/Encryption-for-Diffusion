from sage.all import *
import numpy as np
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler

class BFV():
    def __init__(self, t, q, n):
        '''
        t is the plaintext coefficient, must be prime
        q is the ciphertext coefficient, must be prime
        n is the degree (ring dimension), usually a power of 2
        Usually q >> t
        plaintext ring: P = Z_t[x]/(x^n + 1)
        ciphertext ring: C = R_q X R_q where R_q = Z_q[x]/(x^n + 1)
        '''
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

    def _sample_error_dtbn(self, mu=0, sigma=8/sqrt(2 * pi), beta=19):
        '''
        Draws a random sample from the error distribution (discrete Gaussian) with the given parameters
        '''
        sampler = DiscreteGaussianDistributionIntegerSampler(sigma=sigma, c=mu, tau=beta)
        return sampler()

    def key_gen(self) -> tuple[np.array, PolynomialRing]:
        '''
        Returns (sk, pk, ek)
        sk (secret key) is in R_2
        pk (public key) = (pk1, pk2) is in (R_q, R_q)
        ek (evaluation key) = (ek1, ek2) is in (R_q, R_q)
        '''
        sk = np.array([np.random.choice([-1, 0, 1]) for _ in range(self.n)])
        xbar = self.R_q.gen()
        a = sum([np.random.randint(low=0, high=self.q) * (xbar ** i) for i in range(self.n)])
        sk_q = self.R_q(list(sk))  # sk cast to R_q
        pk1 = -(a*sk_q + self._sample_error_dtbn())
        pk2 = a
        pk = (pk1, pk2)
        ek = (pk1 + sk_q ** 2, pk2)
        return (sk, pk, ek)


    def encrypt(self, pk, m) -> PolynomialRing:
        '''
        Encrypts m (message) into c (its ciphertext) using pk (the public key)
        '''
        pass

    def decrypt(self, sk, c) -> PolynomialRing:
        '''
        Decrypts c (ciphertext) into m (its message) using sk (the secret key)
        '''
        pass
    
    def eval_add(self, c1, c2) -> PolynomialRing:
        '''
        If c1 is an encryption of m1 and c2 is an encryption of m2,
        outputs a ciphertext c3 encrypting (m1 + m2)
        '''
        return (c1[0]+c2[0], c1[1]+c2[1])
    
    def eval_mult(self, c1, c2) -> PolynomialRing:
        '''
        If c1 is an encryption of m1 and c2 an encryption of m2,
        outputs a ciphertext encrypting (m1 * m2)
        '''
        a = self.t*(c1[0]*c2[0])/self.q
        b = self.t*(c1[0]*c2[1] + c1[1]*c2[0])/self.q
        c = self.t*(c1[1]*c2[1])/self.q
        return (a, b, c)

    def relinearize(self, c, ek) -> PolynomialRing:
        return (c[0]+ek*c[2], c[1]+ek*c[2])

n = 4
t = 5
q = 11

bfv = BFV(t, q, n)
(sk, pk, ek) = bfv.key_gen()
print(sk)
print(pk)
print(ek)
