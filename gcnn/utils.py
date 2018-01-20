

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
    
    