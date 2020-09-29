import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
        """
        self.patience = patience
        self.counter = 0
        self.best_accs = None
        self.early_stop = False
        self.F1 = 0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0

    def __call__(self, accs, F1, F2, F3, F4):
        if not self.best_accs:
            self.best_accs = accs
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
        
        if accs < self.best_accs:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_accs = accs
            self.counter = 0
            self.F1 = F1
            self.F2 = F2
            self.F3 = F3
            self.F4 = F4
