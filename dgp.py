import numpy as np


def generate_gaussian_data(n, k, sigma, edge_var=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    I_d = np.eye(k)
    R = np.block([
            [I_d,          sigma * I_d],
            [sigma * I_d,  I_d        ]
        ])
    
    q = np.random.multivariate_normal(np.zeros(2*k), R, size=n)
    Z = q[:, :k]
    X = q[:, k:]

    A = rng.normal(loc=Z @ Z.T, scale=edge_var)
    B = rng.normal(loc=X @ X.T, scale=edge_var)
    # symmetrise
    A = (A + A.T) / 2 
    B = (B + B.T) / 2
    
    return A, B, Z, X