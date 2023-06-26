import numpy as np


def repeat_ratio_batch(samples, preds, k=20):
    """Take the repeat ratio for a single batch.

    Args:
        samples: List-like of sample arrays.
        preds: List-like of prediction arrays.
        k (int, optional): Cut-off point for index ranks. Defaults to 20.

    Returns:
        list: List of repeat ratios per sample
    """
    ratios = []
    
    # go through each sample
    for s_in, s_out in zip(samples, preds):
        topk = np.flip(np.argsort(s_out))[:k] # get topk
        ratio = np.intersect1d(s_in, topk).shape[0] / k
        ratios.append(ratio)

    return ratios


def test0():
    sample = np.array([[0,1,5,6,7,8,9]])
    pred = np.array([[1,1,-1,-1,-1]])
    ratio = repeat_ratio_batch(sample, pred, k=5)[0]
    assert ratio == 0.4, f"Ratio should be 0.4, not {ratio}!"
    print("test0 passed.")

def test1():
    sample = np.array([[1,4,5,6,7,8,9]])
    pred = np.array([[1,1,-1,-1,-1]])
    ratio = repeat_ratio_batch(sample, pred, k=5)[0]
    assert ratio == 0.4, f"Ratio should be 0.4, not {ratio}!"
    print("test1 passed.")


if __name__ == "__main__":
    # run a unit test
    test0()
    test1()