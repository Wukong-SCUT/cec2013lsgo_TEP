from Benchmarks import Benchmarks
import numpy as np
import torch

class F7(Benchmarks):
    def __init__(self,device):
        super().__init__(device)
        self.device = device
        self.ID = 7
        self.s_size = 7
        self.Ovector = self.readOvector().to(self.device)
        self.Pvector = self.readPermVector().to(self.device)
        self.r25 = self.readR(25).to(self.device)
        self.r50 = self.readR(50).to(self.device)
        self.r100 = self.readR(100).to(self.device)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size).to(self.device)
        self.minX = -100.0
        self.maxX = 100.0
        self.r_min_dim = 25
        self.r_med_dim = 50
        self.r_max_dim = 100
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

        c = 0

        self.anotherz = x - self.Ovector

        for i in range(self.s_size):
            anotherz1 = self.rotateVector(i, c)
            anotherz1 = self.transform_osz(anotherz1)
            anotherz1 = self.transform_asy(anotherz1, 0.2)
            result += self.w[i] * self.schwefel(anotherz1)
            c += self.s[i]  # 更新c的值

        if c < self.dimension:
            z = self.anotherz[:, self.Pvector[c:self.dimension]]
            z = self.transform_osz(z)
            z = self.transform_asy(z, 0.2)
            result += self.sphere(z)

        return result
