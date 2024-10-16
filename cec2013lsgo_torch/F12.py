from Benchmarks import Benchmarks
import numpy as np
import torch

class F12(Benchmarks):
    def __init__(self,device):
        super().__init__(device)
        self.device = device
        self.ID = 12
        self.s_size = 20
        self.Ovector = self.readOvector().to(self.device)
        self.minX = -100.0
        self.maxX = 100.0
        self.anotherz = torch.zeros(self.dimension).to(self.device) 

    def __call__(self, x):
        return self.compute(x)

    def info(self):
        info = {'best': 0.0, 'dimension': self.dimension, 'lower': self.minX, 'threshold': 0, 'upper': self.maxX}
        return info

    def compute(self, x):

        # if x is numpy, transform numpy to tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float64)
        x = x.clone().detach().requires_grad_(True) # Convert to tensor
        x.to(self.device)

        if x.ndim == 1:
            x = x.view(1, -1)
        
        result = torch.zeros(x.shape[0]).to(self.device)

        self.anotherz = x - self.Ovector
        result = self.rosenbrock(self.anotherz)
        return result
