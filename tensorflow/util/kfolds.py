from sklearn.model_selection import KFold


def split_k_folds(data, n_splits, random_state, shuffle):
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    return kf.split(data)
