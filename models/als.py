import numpy as np


def solve_side(R, Y, reg, alpha):
    n, f = R.shape[0], Y.shape[1]
    YtY = Y.T @ Y
    X = np.zeros((n, f), dtype=np.float64)
    eye = reg * np.eye(f)
    for u in range(n):
        start, end = R.indptr[u], R.indptr[u + 1]
        if start == end:
            continue
        idx = R.indices[start:end]
        c = alpha * R.data[start:end]
        Yu = Y[idx]
        A = YtY + (Yu.T * c) @ Yu + eye
        b = ((1 + c)[:, None] * Yu).sum(axis=0)
        X[u] = np.linalg.solve(A, b)
    return X


def fit(R, factors=64, reg=1.0, alpha=10.0, iters=10, seed=42, verbose=False):
    rng = np.random.default_rng(seed)
    n_users, n_items = R.shape
    U = rng.normal(0, 0.01, (n_users, factors))
    V = rng.normal(0, 0.01, (n_items, factors))
    Rt = R.T.tocsr()
    for it in range(iters):
        U = solve_side(R, V, reg, alpha)
        V = solve_side(Rt, U, reg, alpha)
        if verbose:
            print('iter', it)
    return U.astype(np.float32), V.astype(np.float32)


def get_scores(model, users):
    U, V = model
    return U[users] @ V.T
