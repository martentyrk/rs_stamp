import numpy as np

def repeat_ratio(samples, preds, k=20):
    ratios = []
    
    # go through each batch
    for b_in, b_out in zip(samples, preds):
        # go through samples in each batch
        for s_in, s_out in zip(b_in, b_out):
            topk = np.flip(np.argsort(s_out))[:k] # get topk
            ratio = np.intersect1d(s_in, topk) / k
            ratios.append(ratio)

    return ratios

def repeat_ratio_sample(samples, preds, k=20):
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
    ratio = repeat_ratio_sample(sample, pred, k=5)[0]
    assert ratio == 0.4, f"Ratio should be 0.4, not {ratio}!"

if __name__ == "__main__":
    # run a unit test
    test0()