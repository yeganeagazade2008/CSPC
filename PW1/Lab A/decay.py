import numpy as np

def simulate(N0, rate, seed=None):
    if rate < 0:
        raise ValueError("Rate cannot be negative")
    if seed is not None:
        np.random.seed(seed)
    
    atoms = N0
    history = [atoms]
    while atoms > 0:
        decayed = np.random.binomial(atoms, rate)
        atoms -= decayed
        history.append(atoms)
        if len(history) > 100:
            break
    return history

def simulate_numpy(N0, rate):
    probs = np.random.rand(N0)
    return np.sum(probs > rate)
