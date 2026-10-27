import numpy as np
import pandas as pd
import torch


def year_bucket(y):
    try:
        y = int(str(y)[:4])
    except ValueError:
        return 'unk'
    if y < 1900:
        return '<1900'
    if y < 1990:
        return '1900-1990'
    if y < 2005:
        return '1990-2005'
    if y < 2012:
        return '2005-2012'
    if y < 2016:
        return '2012-2016'
    return '2016+'


def split_list(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in str(s).split(',') if x.strip() != '']


def pad(rows, max_len):
    idx = np.zeros((len(rows), max_len), dtype=np.int64)
    w = np.zeros((len(rows), max_len), dtype=np.float32)
    for r, row in enumerate(rows):
        for c, (f, val) in enumerate(row[:max_len]):
            idx[r, c] = f
            w[r, c] = val
    return idx, w


def make_features(items, users, item_ids, user_ids, use_meta=True, min_author=2):
    names = {}

    def fid(name):
        if name not in names:
            names[name] = len(names) + 1
        return names[name]

    users = users.set_index('user_id').reindex(user_ids)
    items = items.set_index('id').reindex(item_ids)

    user_rows = []
    for k, (uid, row) in enumerate(users.iterrows()):
        feats = [(fid('user_' + str(k)), 1.0)]
        if use_meta:
            if not pd.isna(row['age']):
                feats.append((fid('age_' + str(row['age'])), 1.0))
            if not pd.isna(row['sex']):
                feats.append((fid('sex_' + str(int(row['sex']))), 1.0))
        user_rows.append(feats)

    author_cnt = pd.Series([a for s in items['authors'] for a in split_list(s)]).value_counts()
    good_authors = set(author_cnt[author_cnt >= min_author].index)

    item_rows = []
    for k, (iid, row) in enumerate(items.iterrows()):
        feats = [(fid('item_' + str(k)), 1.0)]
        if use_meta:
            feats.append((fid('year_' + year_bucket(row['year'])), 1.0))
            genres = split_list(row['genres'])
            for g in genres:
                feats.append((fid('genre_' + g), 1.0 / len(genres)))
            authors = [a for a in split_list(row['authors']) if a in good_authors]
            for a in authors:
                feats.append((fid('author_' + a), 1.0 / len(authors)))
        item_rows.append(feats)

    max_u = max(len(r) for r in user_rows)
    max_i = min(max(len(r) for r in item_rows), 12)
    uidx, uw = pad(user_rows, max_u)
    iidx, iw = pad(item_rows, max_i)
    return {'uidx': uidx, 'uw': uw, 'iidx': iidx, 'iw': iw, 'n_feat': len(names) + 1, 'names': names}


class FM(torch.nn.Module):
    def __init__(self, n_feat, dim):
        super().__init__()
        self.w = torch.nn.Embedding(n_feat, 1, padding_idx=0)
        self.v = torch.nn.Embedding(n_feat, dim, padding_idx=0)
        torch.nn.init.normal_(self.v.weight, 0, 0.01)
        torch.nn.init.zeros_(self.w.weight)
        with torch.no_grad():
            self.v.weight[0] = 0

    def side(self, idx, w):
        lin = (self.w(idx).squeeze(-1) * w).sum(1)
        v = self.v(idx) * w.unsqueeze(-1)
        s = v.sum(1)
        q = (v * v).sum((1, 2))
        return lin, s, q

    def forward(self, uidx, uw, iidx, iw):
        lu, su, qu = self.side(uidx, uw)
        li, si, qi = self.side(iidx, iw)
        return lu + li + 0.5 * ((su + si) ** 2).sum(1) - 0.5 * (qu + qi)


def fit(train, feats, dim=64, reg=1e-4, lr=3e-3, epochs=10, batch=8192, seed=42, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    uidx = torch.tensor(feats['uidx'])
    uw = torch.tensor(feats['uw'])
    iidx = torch.tensor(feats['iidx'])
    iw = torch.tensor(feats['iw'])
    n_items = len(iidx)

    model = FM(feats['n_feat'], dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    users = train['uid'].values
    pos = train['iid'].values

    for ep in range(epochs):
        perm = rng.permutation(len(users))
        neg = rng.integers(0, n_items, len(users))
        total = 0
        for start in range(0, len(perm), batch):
            b = perm[start:start + batch]
            u = torch.tensor(users[b])
            i = torch.tensor(pos[b])
            j = torch.tensor(neg[b])
            lu, su, qu = model.side(uidx[u], uw[u])
            li, si, qi = model.side(iidx[i], iw[i])
            lj, sj, qj = model.side(iidx[j], iw[j])
            diff = li - lj + (su * (si - sj)).sum(1) + 0.5 * ((si ** 2).sum(1) - qi) - 0.5 * ((sj ** 2).sum(1) - qj)
            loss = -torch.nn.functional.logsigmoid(diff).mean()
            l2 = (su ** 2).sum() + (si ** 2).sum() + (sj ** 2).sum()
            loss = loss + reg * l2 / len(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if verbose:
            print('epoch', ep, 'loss', round(total / len(perm), 4))

    model.eval()
    with torch.no_grad():
        li, si, qi = model.side(iidx, iw)
        item_part = li + 0.5 * (si ** 2).sum(1) - 0.5 * qi
    return {'model': model, 'feats': feats, 'si': si.numpy(), 'item_part': item_part.numpy()}


def get_scores(fm, users):
    model, feats = fm['model'], fm['feats']
    with torch.no_grad():
        u = torch.tensor(users)
        lu, su, qu = model.side(torch.tensor(feats['uidx'])[u], torch.tensor(feats['uw'])[u])
        user_part = (lu + 0.5 * (su ** 2).sum(1) - 0.5 * qu).numpy()
    return user_part[:, None] + su.numpy() @ fm['si'].T + fm['item_part'][None, :]
