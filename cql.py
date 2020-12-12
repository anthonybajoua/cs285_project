import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn

from cs285.infrastructure import pytorch_util as ptu

class CQF(nn.Module):

  def __init__(alpha, nS, nD, hidSize, nHidden, lr=1e-3):
    '''
    CQL with hyperparameters alpha, nS, nA and neural net
    parameters as above.
    '''
    super(CQF, self).__init__()
    self.alpha = alpha
    self.net = createMLP(nS, nD, hidSize, nHidden)


  def forward(states):
    '''
    Returns values for all states
    '''
    return self.net(states)

  def update(states, targets, actions):






def createMLP(inSize, outSize, hidSize, nHidden, activation=nn.ReLU()):
    '''
    inSize - input size
    outSize - output size
    hidSize - hidden layer size
    nHidden - number hidden layers
    '''
    activation = nn.ReLU
    bc_loss = nn.MSELoss()
    layers = [nn.Linear(inSize, hidSize)]
    for i in range(nHidden):
        layers.append(nn.Linear(hidSize, hidSize))
        layers.append(activation())
    layers.append(nn.Linear(hidSize, outSize))
    net = nn.Sequential(*layers)
    return net
