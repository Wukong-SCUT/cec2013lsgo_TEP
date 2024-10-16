from Benchmarks import Benchmarks
import numpy as np
import torch

class F2(Benchmarks):
    def __init__(self,device):
        super().__init__(device)
        self.device = device
        self.ID = 2   
        self.Ovector = self.readOvector().to(self.device)
        self.minX = -5.0
        self.maxX = 5.0
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
        result = torch.zeros(x.shape[0])
        
        self.anotherz = x - self.Ovector
        self.anotherz.to(self.device)
        self.anotherz = self.transform_osz(self.anotherz)
        self.anotherz = self.transform_asy(self.anotherz, 0.2)
        self.anotherz = self.Lambda(self.anotherz, 10)
        
        result = self.rastrigin(self.anotherz)
        return result
     