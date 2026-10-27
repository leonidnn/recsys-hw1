import numpy as np


def fit(X, lam=500):
    Xb = X.copy()
    Xb.data[:] = 1
    G = (Xb.T @ Xb).toarray().astype(np.float64)
    n = G.shape[0]
    G[np.arange(n), np.arange(n)] += lam
    P = np.linalg.inv(G)
    del G
    B = P / (-np.diag(P))
    B[np.arange(n), np.arange(n)] = 0
    return B.astype(np.float32)


def get_scores(B, X, users):
    Xb = X[users].copy()
    Xb.data[:] = 1
    return Xb @ B
