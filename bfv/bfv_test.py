import numpy as np
import random
from bfv import BFV

# TODO: Refactor test code to factor out boilerplate (e.g. parameter ranges)

def run_tests(test_fn):
    '''
    test_fn takes bfv, n, param_t, param_q and runs some test
    '''
    for n in [5, 10, 20, 40, 80]:
        for param_t in [5, 10, 15, 20, 40]:
    # for n in [5]:
        # for param_t in [21]:
            param_q = param_t * 2  # Note: change this to explore how param_q affects relative error!
            # In general, as param_q gets larger w.r.t. param_t, relative error gets smaller.
            bfv = BFV(n, param_t, param_q)
            test_fn(bfv, n, param_t, param_q)

def encrypt_then_decrypt_test_fn(bfv, n, param_t, param_q):
    (sk, pk, ek) = bfv.key_gen()
    rng = np.random.default_rng()
    message = rng.integers(low=0, high=bfv.t, size=n)
    m = bfv.array_to_P_ring(message)
    enc = bfv.encrypt(pk, m)
    # print(enc)
    dec = bfv.decrypt(sk, enc)
    # print(dec)

    print("(n, param_t, param_q) =", (n, param_t, param_q))
    err = bfv.diff_P_ring(m, dec)
    # print(err)
    max_abs_err = err.max()
    max_rel_err = max_abs_err / (2 ** param_t)
    print("max absolute error:", max_abs_err)
    print("max relative error (w.r.t. 2^param_t):", max_rel_err)
    
    # Feel free to change this. I'm not sure what the theoretical error should be.
    #assert max_rel_err < max(20 * 2 ** (-param_t), 1e-7), f"original message: {message}\ndecrypted message: {bfv.poly_to_array(dec)}\nerror: {err}"
    print()

def encrypt_then_decrypt():
    print("Running encrypt_then_decrypt...")
    run_tests(encrypt_then_decrypt_test_fn)

def eval_add_const_test_fn(bfv, n, param_t, param_q):
    (sk, pk, ek) = bfv.key_gen()
    m = 42
    m_poly = bfv._polynomialize(m)
    
    ct = bfv.encrypt(pk, m_poly)

    n = 17
    ct_added = bfv.eval_add_const(ct, n, pk, ek)

    decrypted_poly = bfv.decrypt(sk, ct_added)
    sum_poly = bfv._polynomialize(m+n)

    print(f"Original int:  {m}")
    print(f"Constant:      {n}")
    print(f"Expected:      {sum_poly}")
    print(f"Decrypted poly: {decrypted_poly}")

def eval_mult_const_test_fn(bfv, n, param_t, param_q):
    (sk, pk, ek) = bfv.key_gen()
    m = 42
    m_poly = bfv._polynomialize(m)
    
    ct = bfv.encrypt(pk, m_poly)

    n = 17
    ct_added = bfv.eval_mult_const(ct, n, pk, ek)

    decrypted_poly = bfv.decrypt(sk, ct_added)
    sum_poly = bfv._polynomialize(m*n)

    print(f"Original int:  {m}")
    print(f"Constant:      {n}")
    print(f"Expected:      {sum_poly}")
    print(f"Decrypted poly: {decrypted_poly}")

def eval_add_test_fn(bfv, n, param_t, param_q):
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

    # Feel free to change this. I'm not sure what the theoretical error should be.
    assert max_rel_err < max(20 * 2 ** (-param_t), 1e-7), (bfv.poly_to_array(m1 + m2), bfv.poly_to_array(dec), err)
    print()


def eval_mult_test_fn(bfv, n, param_t, param_q):
    (sk, pk, ek) = bfv.key_gen()
    rng = np.random.default_rng()
    message_1 = rng.integers(low=0, high=int(bfv.t / 10), size=n)
    message_2 = rng.integers(low=0, high=int(bfv.t / 10), size=n)
    m1 = bfv.array_to_P_ring(message_1)
    m2 = bfv.array_to_P_ring(message_2)
    enc1 = bfv.encrypt(pk, m1)
    enc2 = bfv.encrypt(pk, m2)
    eval_mult_ans = bfv.eval_mult(ek, enc1, enc2)
    dec = bfv.decrypt(sk, eval_mult_ans)
    
    print("(n, param_t, param_q) =", (n, param_t, param_q))
    # print(bfv.poly_to_array(m1 + m2))
    # print(bfv.poly_to_array(dec))
    # err = np.abs(bfv.poly_to_array(m1 + m2) - bfv.poly_to_array(dec))
    err = bfv.diff_P_ring(m1 * m2, dec)
    # print(err)
    max_abs_err = err.max()
    max_rel_err = max_abs_err / bfv.t
    print("max absolute error:", max_abs_err)
    print("max relative error (w.r.t. 2^param_t):", max_rel_err)
    #print("expected: ", bfv.poly_to_array(m1 + m2))
    #print("actual: ",  bfv.poly_to_array(dec))

    # Feel free to change this. I'm not sure what the theoretical error should be.
    assert max_rel_err < max(20 * 2 ** (-param_t), 1e-7), (bfv.poly_to_array(m1 + m2), bfv.poly_to_array(dec), err)
    print()

def int_encoding_test_fn(bfv, n, param_t, param_q):
    (sk, pk, ek) = bfv.key_gen()
    print(f"n={n}, t={param_t}, q={param_q}")
    for _ in range(100):
        num = random.randint(-2 ** bfv.n + 1, 2 ** bfv.n - 1)
        poly = bfv.encode_int(num)
        dec_num = bfv.decode_int(poly)
        assert num == dec_num, f"expected: {num}, actual: {dec_num}"

def test_eval_add():
    print("Running test_eval_add...")
    run_tests(eval_add_test_fn)

def test_eval_mult():
    print("Running test_eval_mult...")
    # run_tests(eval_mult_test_fn)
    bfv = BFV(2, 5, 10)
    (sk, pk, ek) = bfv.key_gen()
    rng = np.random.default_rng()
    message_1 = [3, 0]
    message_2 = [4, 0]
    m1 = bfv.list_to_P_ring(message_1)
    m2 = bfv.list_to_P_ring(message_2)
    prod = m1 * m2
    enc1 = bfv.encrypt(pk, m1)
    # print(enc1)
    enc2 = bfv.encrypt(pk, m2)
    # print(enc2)
    eval_mult_ans = bfv.eval_mult(ek, enc1, enc2)
    print("enc:", eval_mult_ans)
    dec = bfv.decrypt(sk, eval_mult_ans)
    print("dec:", dec)
    # print(bfv.decrypt(sk, bfv.encrypt(pk, prod)))

def test_eval_add_const():
    print("Running test_eval_add_const...")
    run_tests(eval_add_const_test_fn)

def test_eval_mult_const():
    print("Running test_eval_mult_const...")
    run_tests(eval_mult_const_test_fn)

def test_int_encoding():
    print("Running test_int_encoding...")
    run_tests(int_encoding_test_fn)

def run_all_tests():
    # encrypt_then_decrypt()
    # print("-" * 60)
    # test_eval_add()
    # print("-" * 60)
    # test_eval_add_const()
    # print("-" * 60)
    # test_eval_mult_const()
    test_eval_mult()
    # test_int_encoding()

run_all_tests()
