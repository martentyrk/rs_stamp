import numpy as np

def load_random( word2idx, pad_idx=0, edim=300, init_std=0.05):
    # sigma = np.sqrt(2./(len(word2idx)-1))
    emb_dict = np.random.normal(0, init_std, [len(word2idx), edim])
    # emb_dict = np.random.randn(*(len(word2idx),edim))*sigma
    emb_dict[pad_idx] = [0.0] * edim
    # diag = np.ones(n_items)
    # emb_dict = np.diag(diag)
    return emb_dict

