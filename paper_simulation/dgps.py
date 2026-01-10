import numpy as np

def sbm_data(n, rng=None, correlated=True):
    if rng is None:
        rng = np.random.default_rng()
    
    A = np.zeros((n, n))
    
    Z = rng.multinomial(1, [1/3, 1/3, 1/3], size=n)
    Z_onehot = np.argmax(Z, axis=1)
    
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = np.abs(Z_onehot[i] - Z_onehot[j])
            if diff == 0:
                A[i, j] = rng.binomial(1, 0.5)
            elif diff == 1:
                A[i, j] = rng.binomial(1, 0.2)
            else:
                A[i, j] = rng.binomial(1, 0.4)
            
            A[j, i] = A[i, j]
            
    
    if correlated:
        p = (Z+1)/4
        X = rng.multinomial(1, p)
    else:
        X = rng.multinomial(1, [1/3, 1/3, 1/3], size=n)
    
    return A, X

def generate_gaussian_data(n, k, sigma, edge_var=1, rng=None, link_function='linear'):
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

    
    if link_function == 'linear':
        A = rng.normal(loc=Z @ Z.T, scale=edge_var)
        B = rng.normal(loc=X @ X.T, scale=edge_var)
    elif link_function == 'logistic':
        prob_A = 1 / (1 + np.exp(-(Z @ Z.T)))
        A = rng.binomial(1, prob_A)

        prob_B = 1 / (1 + np.exp(-(X @ X.T)))
        B = rng.binomial(1, prob_B)

    # symmetrise
    A = np.triu(A, k=1)
    A = A + A.T
    B = np.triu(B, k=1)
    B = B + B.T

    return A, B, Z, X

def generate_data(n, type='SBM', rng=None):
    if rng is None:
        rng = np.random.default_rng()
    

    if type == 'SBM':
        return sbm_data(n, rng=rng)
    elif type == 'gaussian':
        return generate_gaussian_data(n, k=2, sigma=0.5, edge_var=1, rng=rng)
    else:
        raise ValueError("Unknown data type")