import pytest
import numpy as np
import decay

def test_simulate_initial():
    res = decay.simulate(1000, 0.4)
    assert res[0] == 1000

def test_negative_rate_raises_error():
    with pytest.raises(ValueError):
        decay.simulate(1000, -0.1)

def test_simulation_average():
    N0 = 1000
    rate = 0.4
    runs = [decay.simulate(N0, rate, seed=i)[1] for i in range(100)]
    avg_remaining = np.mean(runs)
    expected = N0 * (1 - rate)
    assert avg_remaining == pytest.approx(expected, rel=1e-1)
