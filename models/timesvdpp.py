import numpy as np
import torch
import scipy.sparse as sp

MAX_DAY = 730


def get_bin(day, n_bins):
    return np.minimum(np.asarray(day) * n_bins // MAX_DAY, n_bins - 1)


def get_dev(day, mean_day, beta=0.4):
    d = np.asarray(day, dtype=np.float64) - mean_day
    return np.sign(d) * np.abs(d) ** beta


def user_mean_days(train, ratings, n_users):
    mean_all = np.zeros(n_users)
    g = train.groupby('uid')['day'].mean()
    mean_all[g.index.values] = g.values
    g = ratings.groupby('uid')['day'].mean()
    mean_all[g.index.values] = g.values
    return mean_all


def bag_inputs(N, users):
    starts = N.indptr[users]
    ends = N.indptr[users + 1]
    lens = ends - starts
    offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
    idx = np.concatenate([N.indices[s:e] for s, e in zip(starts, ends)])
    w = np.repeat(1 / np.sqrt(np.maximum(lens, 1)), lens)
    return torch.tensor(idx, dtype=torch.long), torch.tensor(offsets, dtype=torch.long), torch.tensor(w, dtype=torch.float32)


class TimeSVDpp(torch.nn.Module):
    def __init__(self, n_users, n_items, n_pairs, n_bins, dim, mu, use_time=True):
        super().__init__()
        self.use_time = use_time
        self.mu = mu
        self.bu = torch.nn.Embedding(n_users, 1)
        self.bi = torch.nn.Embedding(n_items, 1)
        self.au = torch.nn.Embedding(n_users, 1)
        self.but = torch.nn.Embedding(n_pairs + 1, 1, padding_idx=0)
        self.bi_bin = torch.nn.Parameter(torch.zeros(n_items, n_bins))
        self.P = torch.nn.Embedding(n_users, dim)
        self.Ap = torch.nn.Embedding(n_users, dim)
        self.Q = torch.nn.Embedding(n_items, dim)
        self.Y = torch.nn.EmbeddingBag(n_items, dim, mode='sum')
        for e in [self.bu, self.bi, self.au, self.but, self.Ap]:
            torch.nn.init.zeros_(e.weight)
        for e in [self.P, self.Q, self.Y]:
            torch.nn.init.normal_(e.weight, 0, 0.01)

    def user_part(self, u, dev, pair, bag):
        imp = self.Y(bag[0], bag[1], per_sample_weights=bag[2])
        pu = self.P(u) + imp
        bias = self.mu + self.bu(u).squeeze(-1)
        if self.use_time:
            pu = pu + self.Ap(u) * dev.unsqueeze(-1)
            bias = bias + self.au(u).squeeze(-1) * dev + self.but(pair).squeeze(-1)
        return bias, pu

    def forward(self, u, i, b, dev, pair, bag):
        bias, pu = self.user_part(u, dev, pair, bag)
        bias = bias + self.bi(i).squeeze(-1)
        if self.use_time:
            bias = bias + self.bi_bin[i, b]
        return bias + (pu * self.Q(i)).sum(1)

    def reg_loss(self, u, i, b, pair):
        r = (self.P(u) ** 2).sum() + (self.Q(i) ** 2).sum()
        r = r + (self.bu(u) ** 2).sum() + (self.bi(i) ** 2).sum()
        if self.use_time:
            r = r + (self.Ap(u) ** 2).sum() + (self.au(u) ** 2).sum()
            r = r + (self.but(pair) ** 2).sum() + (self.bi_bin[i, b] ** 2).sum()
        return r


def fit(train, n_users, n_items, dim=32, reg=0.01, lr=3e-3, epochs=20, n_bins=30, use_time=True,
        batch=1024, seed=42, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ratings = train[train['rating'].notna()]
    mu = ratings['rating'].mean()

    N = sp.csr_matrix((np.ones(len(train)), (train['uid'].values, train['iid'].values)), shape=(n_users, n_items))
    mean_day = user_mean_days(train, ratings, n_users)

    u_arr = ratings['uid'].values
    i_arr = ratings['iid'].values
    r_arr = ratings['rating'].values.astype(np.float32)
    d_arr = ratings['day'].values
    b_arr = get_bin(d_arr, n_bins)
    dev_arr = get_dev(d_arr, mean_day[u_arr]).astype(np.float32)
    pairs = {}
    pair_arr = np.zeros(len(u_arr), dtype=np.int64)
    for k, (u, d) in enumerate(zip(u_arr, d_arr)):
        if (u, d) not in pairs:
            pairs[(u, d)] = len(pairs) + 1
        pair_arr[k] = pairs[(u, d)]

    model = TimeSVDpp(n_users, n_items, len(pairs), n_bins, dim, mu, use_time)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        perm = rng.permutation(len(u_arr))
        total = 0
        for start in range(0, len(perm), batch):
            b = perm[start:start + batch]
            u = torch.tensor(u_arr[b])
            i = torch.tensor(i_arr[b])
            bb = torch.tensor(b_arr[b])
            pair = torch.tensor(pair_arr[b])
            pred = model(u, i, bb, torch.tensor(dev_arr[b]), pair, bag_inputs(N, u_arr[b]))
            mse = ((pred - torch.tensor(r_arr[b])) ** 2).mean()
            loss = mse + reg * model.reg_loss(u, i, bb, pair) / len(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += mse.item() * len(b)
        if verbose:
            print('epoch', ep, 'train rmse', round(np.sqrt(total / len(perm)), 4))

    model.eval()
    return {'model': model, 'N': N, 'mean_day': mean_day, 'n_bins': n_bins}


def predict_ratings(tsvd, df):
    model = tsvd['model']
    u = df['uid'].values
    dev = get_dev(df['day'].values, tsvd['mean_day'][u]).astype(np.float32)
    b = get_bin(df['day'].values, tsvd['n_bins'])
    with torch.no_grad():
        pred = model(torch.tensor(u), torch.tensor(df['iid'].values), torch.tensor(b), torch.tensor(dev),
                     torch.zeros(len(u), dtype=torch.long), bag_inputs(tsvd['N'], u))
    return pred.numpy()


def get_scores(tsvd, users, day):
    model = tsvd['model']
    dev = get_dev(np.full(len(users), day), tsvd['mean_day'][users]).astype(np.float32)
    b = get_bin(day, tsvd['n_bins'])
    with torch.no_grad():
        u = torch.tensor(users)
        bias, pu = model.user_part(u, torch.tensor(dev), torch.zeros(len(users), dtype=torch.long),
                                   bag_inputs(tsvd['N'], users))
        item_bias = model.bi.weight.squeeze(-1)
        if model.use_time:
            item_bias = item_bias + model.bi_bin[:, int(b)]
        s = bias[:, None] + item_bias[None, :] + pu @ model.Q.weight.T
    return s.numpy()
