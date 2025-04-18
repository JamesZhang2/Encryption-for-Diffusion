import numpy as np
from bfv import BFV

# TODO: Refactor test code to factor out boilerplate (e.g. parameter ranges)

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
            err = bfv.diff_P_ring(m, dec)
            # print(err)
            max_abs_err = err.max()
            max_rel_err = max_abs_err / (2 ** param_t)
            print("max absolute error:", max_abs_err)
            print("max relative error (w.r.t. 2^param_t):", max_rel_err)
            assert max_rel_err < 20 * 2 ** (-param_t)  # Feel free to change this. I'm not sure what the theoretical error should be.

            print()

def test_eval_add():
    print("Running test_eval_add...")
    for n in range(5, 51, 5):
        for param_t in range(5, 21, 5):
            param_q = param_t * 2  # Note: change this to explore how param_q affects relative error!
            # In general, as param_q gets larger w.r.t. param_t, relative error gets smaller.
            bfv = BFV(n, param_t, param_q)
            (sk, pk, ek) = bfv.key_gen()
            rng = np.random.default_rng()
            message_1 = rng.integers(low=0, high=bfv.t, size=n)
            message_2 = rng.integers(low=0, high=bfv.t, size=n)
            m1 = bfv.array_to_P_ring(message_1)
            m2 = bfv.array_to_P_ring(message_2)
            enc1 = bfv.encrypt(pk, m1)
            enc2 = bfv.encrypt(pk, m2)
            eval_add_ans = bfv.eval_add(ek, enc1, enc2)
            dec = bfv.decrypt(sk, eval_add_ans)
            
            print("(n, param_t, param_q) =", (n, param_t, param_q))
            # print(bfv.poly_to_array(m1 + m2))
            # print(bfv.poly_to_array(dec))
            # err = np.abs(bfv.poly_to_array(m1 + m2) - bfv.poly_to_array(dec))
            err = bfv.diff_P_ring(m1 + m2, dec)
            # print(err)
            max_abs_err = err.max()
            max_rel_err = max_abs_err / (2 ** param_t)
            print("max absolute error:", max_abs_err)
            print("max relative error (w.r.t. 2^param_t):", max_rel_err)
            assert max_rel_err < 20 * 2 ** (-param_t)  # Feel free to change this. I'm not sure what the theoretical error should be.

            print()

def run_all_tests():
    encrypt_then_decrypt()
    print("-" * 60)
    test_eval_add()

run_all_tests()