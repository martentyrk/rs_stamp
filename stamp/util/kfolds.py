from sklearn.model_selection import KFold


def split_k_folds(data, n_splits, random_state=None, shuffle=False):
    """
    Split data into n splits/folds

    Outputs:
        List of data split into n folds
    """
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    folds = [fold for fold in kf.split(data)]
    return folds
