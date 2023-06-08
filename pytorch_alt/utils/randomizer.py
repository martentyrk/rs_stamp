import torch
class Randomizer(object):
    stddev = None

    @staticmethod
    def random_normal(wshape):
        return torch.normal(torch.zeros_like(wshape), Randomizer.stddev)
        #return tf.random.normal(wshape, stddev=Randomer.stddev)

    @staticmethod
    def set_stddev(sd):
        Randomizer.stddev = sd