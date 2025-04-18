import numpy as np
from bfv import BFV


def encrypt_then_decrypt():
    print("Running encrypt_then_decrypt...")
    for n in range(5, 51, 5):
        for param_t in range(5, 21, 5):
            param_q = param_t * 2  # Note: change this to explore how param_q affects relative error!
            # In general, as param_q gets larger w.r.t. param_t, relative error gets smaller.
            bfv = BFV(n, param_t, param_q)
            (sk, pk, ek) = bfv.key_gen()
            rng = np.random.default_rng()
            message = rng.integers(low=0, high=bfv.t, size=n)
            m = bfv.array_to_P_ring(message)
            enc = bfv.encrypt(pk, m)
            dec = bfv.decrypt(sk, enc)

            print("(n, param_t, param_q) =", (n, param_t, param_q))
            err = np.abs(message - bfv.poly_to_array(dec))
            # print(err)
            max_abs_err = err.max()
            max_rel_err = max_abs_err / (2 ** param_t)
            print("max absolute error:", max_abs_err)
            print("max relative error (w.r.t. 2^param_t):", max_rel_err)
            assert max_rel_err < 20 * 2 ** (-param_t)  # Feel free to change this. I'm not sure what the theoretical error should be.

            print()
            # np.testing.assert_allclose(np.array(list(dec), dtype=int), message, atol=2)

def run_all_tests():
    encrypt_then_decrypt()

run_all_tests()