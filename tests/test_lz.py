import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain import lz

def test_lempel_ziv_example():
    assert lz.lz76([int(c) for c in '0001101001000101']) == 6

def test_constant_and_periodic_strings_are_simple():
    assert lz.lz76(np.zeros(10000)) == 2
    assert lz.lz76(np.tile([0, 1], 5000)) == 3

def test_random_string_normalised_complexity_near_one():
    rng = np.random.default_rng(0)
    S = rng.integers(0, 2, (200, 500))
    pci = lz.pci_lz(S)
    assert 0.9 < pci < 1.1, pci

def test_linear_matches_kaspar_schuster_reference():
    rng = np.random.default_rng(1)
    for n in (1, 2, 3, 17, 100, 1000, 20000):
        for p1 in (0.5, 0.1, 0.01, 0.001):
            x = (rng.random(n) < p1).astype(np.uint8)
            assert lz.lz76(x) == lz.lz76_quadratic(x), (n, p1)
    for pat in ('0001101001000101', '1' * 50, '01' * 40 + '1', '0' * 999 + '1'):
        x = [int(c) for c in pat]
        assert lz.lz76(x) == lz.lz76_quadratic(x), pat

def test_linear_is_fast_on_a_sparse_benchmark_sized_matrix():
    import time
    rng = np.random.default_rng(2)
    S = (rng.random((166700, 300)) < 0.002).astype(np.uint8)
    t = time.time()
    lz.pci_lz(S)
    assert time.time() - t < 60, f'took {time.time() - t:.0f} s'
if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print('PASS', name)
            except AssertionError as e:
                fails += 1
                print('FAIL', name, '-', e)
    sys.exit(1 if fails else 0)
