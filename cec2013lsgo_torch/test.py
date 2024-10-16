from cec2013 import Benchmark
import torch
import numpy as np

from cec2013lsgo.cec2013 import Benchmark as Benchmark_

bench = Benchmark('cuda')

ID = 15
fun = bench.get_function(ID)
x = torch.rand((3,1000)).to('cuda')
Ovector = fun.readOvector()
Ovector = Ovector.clone().detach()
Ovector = Ovector.to('cuda')
print(fun(Ovector))
print(fun(x))

bench_ = Benchmark_()
fun_ = bench_.get_function(ID)
print(fun_(Ovector.cpu().numpy().astype(np.float64)))
print(fun_(x[0].cpu().numpy().astype(np.float64)))

