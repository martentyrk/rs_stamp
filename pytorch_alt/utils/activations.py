import torch

def activation(inputs, case='tanh'):
    '''
    The active enter. 
    '''
    switch = {
        'tanh': tanh,
        'relu': relu,
        'sigmoid': sigmoid,
    }
    func = switch.get(case, tanh)
    return func(inputs)


def tanh(inputs):
    '''
    The tanh active. 
    '''
    return torch.tanh(inputs)


def sigmoid(inputs):
    '''
    The sigmoid active.
    '''
    return torch.sigmoid(inputs)


def relu(inputs):
    '''
    The relu active. 
    '''
    relu = torch.nn.ReLU(inputs)
    return relu(inputs)