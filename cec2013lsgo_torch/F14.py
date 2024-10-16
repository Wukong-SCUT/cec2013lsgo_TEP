from Benchmarks import Benchmarks
import numpy as np
import torch

class F14(Benchmarks):
    def __init__(self,device):
        super().__init__(device)
        self.device = device
        self.ID = 14
        self.s_size = 20
        self.dimension = 905 #because of overlapping
        self.overlap = 5
        self.s = self.readS(self.s_size)
        self.OvectorVec = self.readOvectorVec()
        self.Pvector = self.readPermVector().to(self.device)
        self.r25 = self.readR(25).to(self.device)
        self.r50 = self.readR(50).to(self.device)
        self.r100 = self.readR(100).to(self.device)
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

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.clone().detach().requires_grad_(True) # Convert to tensor
        x.to(self.device)
        if x.ndim == 1:
            x = x.view(1, -1)

        result = torch.zeros(x.size(0)).to(self.device)

        c=0

        for i in range(self.s_size):
            anotherz1 = self.rotateVectorConflict(i, c, x)
            anotherz1 = self.transform_osz(anotherz1)
            anotherz1 = self.transform_asy(anotherz1, 0.2)
            result += self.w[i] * self.schwefel(anotherz1)
            c += self.s[i]  # 更新c的值

        return result
    


