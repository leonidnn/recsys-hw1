import numpy as np
import scipy.sparse as sp


def cosine_sims(X, shrink):
    Xb = X.copy()
    Xb.data[:] = 1
    co = (Xb.T @ Xb).toarray().astype(np.float32)
    cnt = np.sqrt(np.diag(co)).copy()
    co /= (np.outer(cnt, cnt) + shrink + 1e-8)
    np.fill_diagonal(co, 0)
    return co


def keep_neighbors(S, k):
    n = S.shape[0]
    idx = np.argpartition(-S, k, axis=0)[:k]
    cols = np.tile(np.arange(n), k)
    rows = idx.ravel()
    vals = S[rows, cols]
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    return W


def fit(X, k=100, shrink=0, S=None):
    if S is None:
        S = cosine_sims(X, shrink)
    W = keep_neighbors(S, k)
    return W


def get_scores(W, X, users):
    Xb = X[users].copy()
    Xb.data[:] = 1
    return (Xb @ W).toarray()
