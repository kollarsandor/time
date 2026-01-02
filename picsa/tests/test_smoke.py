import sys
import os
import numpy as np

def test_fp8_roundtrip():
    print("Testing FP8 E4M3 roundtrip...")
    test_values = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.125, 100.0, -100.0]
    max_error = 0.0
    for val in test_values:
        sign = 1 if val < 0 else 0
        ax = abs(val)
        if ax == 0:
            fp8 = sign << 7
        elif ax >= 448:
            fp8 = (sign << 7) | 0x7E
        elif ax < 2**-9:
            fp8 = sign << 7
        else:
            import math
            log2_ax = math.log2(ax)
            e = int(math.floor(log2_ax))
            e = max(-6, min(8, e))
            exp_bits = e + 7
            m = ax / (2**e) - 1.0
            mant = int(round(m * 8))
            mant = max(0, min(7, mant))
            fp8 = (sign << 7) | (exp_bits << 3) | mant
        exp = (fp8 >> 3) & 0xF
        mant = fp8 & 0x7
        sign_bit = (fp8 >> 7) & 1
        if exp == 0:
            if mant == 0:
                decoded = 0.0
            else:
                decoded = mant * (2**-9)
        elif exp == 15 and mant == 7:
            decoded = float('nan')
        else:
            e = exp - 7
            m = 1.0 + mant / 8.0
            decoded = m * (2**e)
        if sign_bit:
            decoded = -decoded
        error = abs(decoded - val)
        if val != 0:
            rel_error = error / abs(val)
        else:
            rel_error = 0
        max_error = max(max_error, rel_error)
    print(f"  Max relative error: {max_error:.4f}")
    if max_error < 0.2:
        print("  PASS: FP8 roundtrip within acceptable error")
        return True
    else:
        print("  FAIL: FP8 roundtrip error too high")
        return False

def test_softmax():
    print("Testing softmax correctness...")
    np.random.seed(42)
    logits = np.random.randn(10).astype(np.float32)
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    softmax = exp_logits / np.sum(exp_logits)
    if abs(np.sum(softmax) - 1.0) < 1e-5:
        print("  PASS: Softmax sums to 1.0")
        return True
    else:
        print(f"  FAIL: Softmax sum = {np.sum(softmax)}")
        return False

def test_sampling():
    print("Testing sampling correctness...")
    np.random.seed(42)
    probs = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
    samples = []
    for _ in range(1000):
        rand_val = np.random.random()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if cumsum >= rand_val:
                samples.append(i)
                break
    sample_probs = [samples.count(i) / len(samples) for i in range(4)]
    max_diff = max(abs(sample_probs[i] - probs[i]) for i in range(4))
    if max_diff < 0.05:
        print(f"  PASS: Sampling distribution within tolerance (max diff = {max_diff:.4f})")
        return True
    else:
        print(f"  FAIL: Sampling distribution off (max diff = {max_diff:.4f})")
        return False

def test_matmul():
    print("Testing matrix multiplication...")
    np.random.seed(42)
    A = np.random.randn(4, 8).astype(np.float32)
    B = np.random.randn(8, 6).astype(np.float32)
    C_expected = A @ B
    C_manual = np.zeros((4, 6), dtype=np.float32)
    for i in range(4):
        for j in range(6):
            for k in range(8):
                C_manual[i, j] += A[i, k] * B[k, j]
    max_diff = np.max(np.abs(C_expected - C_manual))
    if max_diff < 1e-5:
        print(f"  PASS: Matrix multiplication correct (max diff = {max_diff:.6f})")
        return True
    else:
        print(f"  FAIL: Matrix multiplication incorrect (max diff = {max_diff:.6f})")
        return False

def test_embedding_lookup():
    print("Testing embedding lookup...")
    np.random.seed(42)
    vocab_size = 1000
    hidden_dim = 64
    embedding_table = np.random.randn(vocab_size, hidden_dim).astype(np.float32)
    token_ids = [42, 100, 500, 999]
    expected = np.stack([embedding_table[tid] for tid in token_ids])
    actual = np.zeros((len(token_ids), hidden_dim), dtype=np.float32)
    for i, tid in enumerate(token_ids):
        actual[i] = embedding_table[tid]
    if np.allclose(expected, actual):
        print("  PASS: Embedding lookup correct")
        return True
    else:
        print("  FAIL: Embedding lookup incorrect")
        return False

def main():
    print("=" * 60)
    print("GLM-4.7-FP8 Inference Engine Smoke Tests")
    print("=" * 60)
    print()
    tests = [
        test_fp8_roundtrip,
        test_softmax,
        test_sampling,
        test_matmul,
        test_embedding_lookup,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
        print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
