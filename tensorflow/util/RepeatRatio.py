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

            


def cau_recall_mrr(preds,labels,cutoff):
    recall = []
    mrr = []
    for batch, b_label in zip(preds,labels): # batch predictions and batch labels
        for step, s_label in zip(batch,b_label): # sample prediction and sample label
            ranks = (step[s_label] < step).sum() + 1 # number of items with probability higher than desired item + 1
            recall.append(ranks<=cutoff) # 1 if desired item rank is under the cutoff
            mrr.append(1/ranks if ranks <= cutoff else 0.0) # 
    return recall, mrr


def cau_samples_recall_mrr(samples, cutoff=20):
    recall = 0.0
    mrr =0.0
    for sample in samples:
        recall += sum(x <= cutoff for x in sample.pred)
        mrr += sum(1/x if x <= cutoff else 0 for x in sample.pred)
    num = 0
    for sample in samples:
        num += len(sample.pred)
    recall = recall/ num
    mrr = mrr/num
    return recall , mrr

