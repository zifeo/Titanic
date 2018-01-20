import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from skorch import NeuralNet
from skorch.utils import to_numpy
from sklearn.metrics import log_loss

from gcnn.layers import GraphChebyConv

conv1_dim = 32
conv2_dim = 64

class Net1(nn.Module):
    def __init__(self):
        super().__init__()

        self.gc1 = nn.Sequential(
            # GraphFourierConv(f0.cuda() if cuda else f, 1, 10, bias=True),
            GraphChebyConv(l0.cuda() if cuda else l0, 1, 10, 20, bias=True),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(len(f0) * 10, 10),
            nn.ReLU(),
            nn.Softmax(1)
        )

    def forward(self, x):
        out = self.gc1(x)
        out = self.fc(out.view(out.size(0), -1))
        return out

class Net2(nn.Module):
    def __init__(self):
        super().__init__()

        self.gc1 = nn.Sequential(
            # GraphFourierConv(l0.cuda() if cuda else l0, 1, conv1_dim, bias=True),
            GraphChebyConv(l0.cuda() if cuda else l0, 1, conv1_dim, 25, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.gc2 = nn.Sequential(
            # GraphFourierConv(l2.cuda() if cuda else l2, conv1_dim, conv2_dim, bias=True),
            GraphChebyConv(l2.cuda() if cuda else l2, conv1_dim, conv2_dim, 25, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.fc = nn.Sequential(
            nn.Linear(len(f2) // 4 * conv2_dim, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 10),
            nn.ReLU(),
            nn.Softmax(1)
        )

    def forward(self, x):
        out = self.gc1(x)
        out = self.gc2(out)
        out = self.fc(out.view(out.size(0), -1))
        return out
    
class NNplusplus(NeuralNet):
    '''
    inherit NeuralNet class from skorch
    '''
    def score(self,X,target):
        '''
        redefine scoring method to be the same as the one of kaggle (log_loss)
        '''
        y_preds = []
        for yp in self.forward_iter(X, training=False):
            y_preds.append(to_numpy(yp.sigmoid()))   
        y_preds = np.concatenate(y_preds, 0)
        return log_loss(target,y_preds)