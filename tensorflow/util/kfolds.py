from sklearn.model_selection import KFold


def split_k_folds(data, n_splits, random_state=None, shuffle=False):
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    folds = [fold for fold in kf.split(data)]
    return folds
