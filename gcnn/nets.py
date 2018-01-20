import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from skorch import NeuralNet
from skorch.utils import to_numpy
from sklearn.metrics import log_loss

from gcnn.layers import GraphChebyConv


class BaselineCNN(nn.Module):
    def __init__(self, conv1_dim=16, conv2_dim=32):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, conv1_dim, kernel_size=7, stride=1, padding=2, groups=2),
            #nn.BatchNorm2d(outsize1),
            nn.ReLU(),
            nn.MaxPool2d(4)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(conv1_dim, conv2_dim, kernel_size=5, stride=1, padding=2),
            #nn.BatchNorm2d(outsize2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        

        self.fc = nn.Sequential(
            nn.Linear(conv2_dim * 9 * 9, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    
class PaperSimpleGC(nn.Module):
    """
    GC10
    """
    
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

    
class PaperGCFC(nn.Module):
    """
    GC32-P4-GC64-P4-FC512
    """
    
    def __init__(self, conv1_dim=32, conv2_dim=64):
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
    
    