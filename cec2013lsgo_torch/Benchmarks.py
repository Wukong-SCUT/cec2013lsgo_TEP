import numpy as torch
import warnings
import sys
import random
import torch
import math



class Benchmarks:
    def __init__(self, device):

        torch.set_default_dtype(torch.float64)
        self.data_dir = "cec2013lsgo/cec2013lsgo_torch/cdatafiles" # 数据文件夹
        self.dimension = 1000 # 维度

        # 子空间的维度大小, 先提供了三种子空间的维度大小
        self.min_dim = 25
        self.med_dim = 50
        self.max_dim = 100  

        # 基本量的设置, 不是准确的值，准确的值会在function中设置
        self.device = device
        self.ID = None
        self.s_size = 20
        self.overlap = None
        self.minX = None
        self.maxX = None
        self.Ovector = None
        self.OvectorVec = None
        self.Pvector = None
        self.r_min_dim = None
        self.r_med_dim = None
        self.r_max_dim = None
        self.anotherz = torch.zeros(self.dimension)
        self.anotherz1 = None
        self.best = 0
    
        self.maxevals = 3000000 #最大评估次数
        self.numevals = 0 # 当前评估次数, 用于记录当前评估次数, cpp中设置的值是2*self.maxevals

        self.output = ""
        self.output_dir = f'cec2013lsgo_py'
        # self.record_evels = torch.array([120000, 600000, 3000000])

    # 读取Ovector
    def readOvector(self):
        d = torch.zeros(self.dimension)
        file_path = f"{self.data_dir}/F{self.ID}-xopt.txt"
        
        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    values = line.strip().split(',')
                    for value in values:
                        if c < self.dimension:
                            d[c] = float(value)
                            c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        
        return d
    
    # 读取OvectorVec，根据子空间的大小分割，得到一个向量数组
    def readOvectorVec(self):
        d = [torch.zeros(self.s[i], device=self.device) for i in range(self.s_size)]
        file_path = f"{self.data_dir}/F{self.ID}-xopt.txt"

        try:
            with open(file_path, 'r') as file:
                c = 0  # index over 1 to dim
                i = -1  # index over 1 to s_size
                up = 0  # current upper bound for one group

                for line in file:
                    if c == up:  # out (start) of one group
                        i += 1
                        d[i] = torch.zeros(self.s[i], device=self.device)
                        up += self.s[i]

                    values = line.strip().split(',')
                    for value in values:
                        d[i][c - (up - self.s[i])] = float(value)
                        c += 1
        except FileNotFoundError:
            print(f"Cannot open the OvectorVec datafiles '{file_path}'")

        return d
    
    # 读取PermVector
    def readPermVector(self):
        d = torch.zeros(self.dimension, dtype=int)
        file_path = f"{self.data_dir}/F{self.ID}-p.txt"
        
        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    values = line.strip().split(',')
                    for value in values:
                        if c < self.dimension:
                            d[c] = int(float(value)) - 1
                            c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        
        return d
    
        # 读取R，即为各个子空间的向量
    def readR(self, sub_dim):
        m = torch.zeros((sub_dim, sub_dim))
        file_path = f"{self.data_dir}/F{self.ID}-R{sub_dim}.txt"

        try:
            with open(file_path, 'r') as file:
                i = 0
                for line in file:
                    values = line.strip().split(',')
                    for j, value in enumerate(values):
                        m[i, j] = float(value)
                    i += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        
        return m

    # 读取S，即为各个子问题的维度
    def readS(self, num):
        self.s = torch.zeros(num, dtype=int)
        file_path = f"{self.data_dir}/F{self.ID}-s.txt"

        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    self.s[c] = int(float(line.strip()))
                    c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        
        return self.s

    # 读取W
    def readW(self, num):
        self.w = torch.zeros(num)
        file_path = f"{self.data_dir}/F{self.ID}-w.txt"

        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    self.w[c] = float(line.strip())
                    c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        
        return self.w

    
    # 向量乘矩阵
    def multiply(self, vector, matrix):
        return torch.matmul(matrix, vector.T).T

    # 旋转向量
    def rotateVector(self, i, c): 
        # 获取子问题的维度
        sub_dim = self.s[i]
        # 将值复制到新向量中
        indices = self.Pvector[c:c + sub_dim]
        z = self.anotherz[:,indices]
        # 选择正确的旋转矩阵并进行乘法运算
        if sub_dim == self.r_min_dim:
            self.anotherz1 = self.multiply(z, self.r25)
        elif sub_dim == self.r_med_dim:
            self.anotherz1 = self.multiply(z, self.r50)
        elif sub_dim == self.r_max_dim:
            self.anotherz1 = self.multiply(z, self.r100)
        else:
            print("Size of rotation matrix out of range")
            self.anotherz1 = None

        return self.anotherz1
    
    def rotateVectorConform(self, i, c):
        sub_dim = self.s[i]
        start_index = c - i * self.overlap
        end_index = c + sub_dim - i * self.overlap
        # 将值复制到新向量中
        indices = self.Pvector[start_index:end_index]
        z = self.anotherz[:, indices]
        # 选择正确的旋转矩阵并进行乘法运算
        if sub_dim == self.r_min_dim:
            self.anotherz1 = self.multiply(z, self.r25)
        elif sub_dim == self.r_med_dim:
            self.anotherz1 = self.multiply(z, self.r50)
        elif sub_dim == self.r_max_dim:
            self.anotherz1 = self.multiply(z, self.r100)
        else:
            print("Size of rotation matrix out of range")
            self.anotherz1 = None
    
        return self.anotherz1

    def rotateVectorConflict(self, i, c, x):
        sub_dim = self.s[i]
        start_index = c - i * self.overlap
        end_index = c + sub_dim - i * self.overlap

        # 将值复制到新向量中并进行减法运算
        indices = self.Pvector[start_index:end_index]
        z = x[:,indices] - self.OvectorVec[i]
        z = z.to(self.device)
        # 选择正确的旋转矩阵并进行乘法运算
        if sub_dim == self.r_min_dim:
            self.anotherz1 = self.multiply(z, self.r25)
        elif sub_dim == self.r_med_dim:
            self.anotherz1 = self.multiply(z, self.r50)
        elif sub_dim == self.r_max_dim:
            self.anotherz1 = self.multiply(z, self.r100)
        else:
            print("Size of rotation matrix out of range")
            self.anotherz1 = None

        return self.anotherz1
    
    # basic function
    def sphere(self,x):
        s2 = torch.sum(x ** 2,axis=-1)
        return s2

    def elliptic(self,x):
        nx = x.shape[-1]
        i = torch.arange(nx).to(x.device)
        return torch.sum(10 ** (6 * i / (nx - 1)) * (x ** 2), -1)

    def rastrigin(self,x):
        return torch.sum(x**2 - 10 * torch.cos(2*torch.pi*x) + 10, -1)

    def ackley(self,x):
        nx = x.shape[-1]
        sum1 = -0.2 * torch.sqrt(torch.sum(x ** 2, -1) / nx)
        sum2 = torch.sum(torch.cos(2 * torch.pi * x), -1) / nx
        return - 20 * torch.exp(sum1) - torch.exp(sum2)+20 +torch.e 

    def schwefel(self,x):
        s1 = torch.cumsum(x,axis=-1)
        s2 = torch.sum(s1 ** 2,axis=-1)
        return s2

    def rosenbrock(self,x):
        x0 = x[:,:x.size(1)-1]
        x1 = x[:,1:x.size(1)]
        t = x0**2 - x1
        s = torch.sum(100.0 * t**2 + (x0 - 1.0)**2,-1)
        return s
    
    def transform_osz(self,z):
        sign_z = torch.sign(z)
        hat_z = torch.where(z == 0, 0, torch.log(torch.abs(z)))
        c1_z = torch.where(z > 0, 10, 5.5)
        c2_z = torch.where(z > 0, 7.9, 3.1)
        sin_term = torch.sin(c1_z * hat_z) + torch.sin(c2_z * hat_z)
        z_transformed = sign_z * torch.exp(hat_z + 0.049 * sin_term)
        return z_transformed

    def transform_asy(self,z, beta=0.2):
        indices = torch.arange(z.shape[-1])[None,:].repeat(z.shape[0], 1).to(self.device)
        positive_mask = z > 0
        z[positive_mask] = z[positive_mask] ** (1 + beta * indices[positive_mask] / (z.shape[-1] - 1) * torch.sqrt(z[positive_mask]))
        return z
    
    def Lambda(self, z, alpha=10):
        dim = z.shape[-1]
        # 创建指数数组
        exponents = 0.5 * torch.arange(dim).to(self.device) / (dim - 1)
        # 计算变换后的z
        z = z * (alpha ** exponents)
        return z
    
    def update(self, newfitness):
        if self.numevals > self.maxevals:
            if self.numevals >= 2 * self.maxevals:
                print("Error: nextRun was not run before compute")
                sys.exit(1)
            elif self.numevals > self.maxevals * 1.1:
                print("Error: many evaluations greater than maximum.")
                sys.exit(1)
            print("Warning: evaluations greater than maximum, will be ignored.")
            return

        if self.numevals == 0 or newfitness < self.best_fitness:
            self.best_fitness = newfitness
            if not self.output:
                self.output = f"results_f{self.ID}.csv"

        self.numevals += 1

        if self.numevals in self.record_evels:
            self.save_evals()
        
    def save_evals(self):
        with open(self.output, 'a') as f_output:
            f_output.write(f"{self.numevals}, {self.ID}, {self.best_fitness:.6e}\n")






    




    






