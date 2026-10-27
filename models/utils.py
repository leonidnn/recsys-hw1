import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_data(path):
    inter = pd.read_csv(path + '/interactions.csv')
    items = pd.read_csv(path + '/items.csv')
    users = pd.read_csv(path + '/users.csv')
    return inter, items, users


def drop_dups(df):
    df = df.sort_values(['progress', 'start_date'])
    df = df.drop_duplicates(['user_id', 'item_id'], keep='last')
    return df


def filter_core(df, min_user, min_item):
    while True:
        cnt = df['item_id'].value_counts()
        df = df[df['item_id'].isin(cnt[cnt >= min_item].index)]
        cnt = df['user_id'].value_counts()
        new_df = df[df['user_id'].isin(cnt[cnt >= min_user].index)]
        if len(new_df) == len(df):
            return new_df
        df = new_df


def encode_ids(df):
    user_ids = np.sort(df['user_id'].unique())
    item_ids = np.sort(df['item_id'].unique())
    df = df.copy()
    df['uid'] = np.searchsorted(user_ids, df['user_id'].values)
    df['iid'] = np.searchsorted(item_ids, df['item_id'].values)
    df['day'] = (pd.to_datetime(df['start_date']) - pd.Timestamp('2018-01-01')).dt.days
    return df, user_ids, item_ids


def time_split(df, date_from, date_to):
    train = df[df['start_date'] < date_from]
    test = df[(df['start_date'] >= date_from) & (df['start_date'] < date_to)]
    return train, test


def make_matrix(df, n_users, n_items, values=None):
    if values is None:
        values = np.ones(len(df), dtype=np.float32)
    X = sp.csr_matrix((values, (df['uid'].values, df['iid'].values)), shape=(n_users, n_items), dtype=np.float32)
    return X


def get_targets(train, test):
    known_users = set(train['uid'].unique())
    known_items = set(train['iid'].unique())
    test = test[test['uid'].isin(known_users) & test['iid'].isin(known_items)]
    seen = set(zip(train['uid'].values, train['iid'].values))
    mask = [(u, i) not in seen for u, i in zip(test['uid'].values, test['iid'].values)]
    test = test[mask]
    targets = test.groupby('uid')['iid'].apply(set).to_dict()
    users = np.array(sorted(targets.keys()))
    return users, targets


def top_k(scores, k):
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    part = np.take_along_axis(scores, top, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(top, order, axis=1)


def recommend(score_fn, users, X_seen, k=10, batch=2000):
    res = []
    for start in range(0, len(users), batch):
        u = users[start:start + batch]
        s = np.array(score_fn(u), dtype=np.float32)
        rows, cols = X_seen[u].nonzero()
        s[rows, cols] = -np.inf
        res.append(top_k(s, k))
    return np.vstack(res)


def calc_metrics(recs, users, targets, n_items, k=10):
    disc = 1 / np.log2(np.arange(2, k + 2))
    ndcg, recall, hit = [], [], []
    for row, u in zip(recs, users):
        true = targets[u]
        rel = np.array([i in true for i in row[:k]], dtype=float)
        idcg = disc[:min(len(true), k)].sum()
        ndcg.append((rel * disc).sum() / idcg)
        recall.append(rel.sum() / len(true))
        hit.append(float(rel.sum() > 0))
    res = {
        'ndcg@10': np.mean(ndcg),
        'recall@10': np.mean(recall),
        'hitrate@10': np.mean(hit),
        'coverage@10': len(np.unique(recs[:, :k])) / n_items,
    }
    return res
