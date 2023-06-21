import numpy as np

def repeat_ratio(samples, preds, k=20):
    ratios = []
    
    # go through each batch
    for b_in, b_out in zip(samples, preds):
        # go through samples in each batch
        for s_in, s_out in zip(b_in, b_out):
            topk = np.argsort(s_out).flip()[:k] # get topk
            ratio = np.intersect1d(s_in, topk) / k
            ratios.append(ratio)

    return ratios