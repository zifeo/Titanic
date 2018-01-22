from skorch import NeuralNet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from timeit import default_timer as timer
import numpy as np 
from skorch.utils import to_numpy
from sklearn.metrics import log_loss


def score_classification(truth, predicted):
    """
    Score according to accuracy, precision, recall and f1.
    """
    print(classification_report(truth, predicted, target_names=['ship', 'iceberg']))
    return [
        accuracy_score(truth, predicted),
        precision_score(truth, predicted),
        recall_score(truth, predicted),
        f1_score(truth, predicted)
    ]


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
    
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

    
from torch.autograd import Variable
import torch as th
from collections import OrderedDict


def summary(input_size, model):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

                params = 0
                if hasattr(module, 'weight'):
                    params += th.prod(th.LongTensor(list(module.weight.size())))
                    if module.weight.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False
                if hasattr(module, 'bias'):
                    params +=  th.prod(th.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = params
                
            if(not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
                hooks.append(module.register_forward_hook(hook))
        
        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(th.rand(1,*in_size)) for in_size in input_size]
        else:
            x = Variable(th.rand(1,*input_size))

        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        #model.apply(register_hook)
        # make a forward pass
        model(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        return summary
    