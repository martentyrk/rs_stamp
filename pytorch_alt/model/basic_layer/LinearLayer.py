#coding=utf-8
import torch
from utils.randomizer import Randomizer

class LinearLayer(object):

    def __init__(self, w_shape, stddev = None, params=None):
        '''
        :param w_shape: [input_dim, output_dim]
        :param stddev: 用于初始化
        :param params: 从外界制定参数
        '''
        if params is None:

            #self.w = tf.Variable(
            #    Randomizer.random_normal(w_shape),
            #    trainable=True
            #)

            self.w = torch.nn.Parameter(
                torch.Tensor(Randomizer.random_normal(w_shape)), 
                requires_grad=True
            )
        else:
            self.w = params['w']
    def forward(self, inputs):
        '''
        count
        '''
        #res = tf.matmul(inputs, self.w)
        res = torch.matmul(inputs, self.w)
        return res