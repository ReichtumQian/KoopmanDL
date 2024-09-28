import torch
import KoopmanDL as kd


def func(x):
  N = x.size(0)
  M = 3
  result = torch.zeros(N, M)
  for i in range(N):
    result[i] = torch.tensor(
      [1, x[i], x[i]**2]
    )
  return result
    
data_x = torch.tensor(
  [[1], [2], [3]], dtype=torch.float
)
data_y = torch.tensor(
  [[2], [3], [4]], dtype=torch.float
)
M = 3
dic = kd.Dictionary(M, func)
solver = kd.EDMDSolver(dic)
K = solver.compute_K(data_x, data_y)
print(K)
layer = torch.nn.Linear(M, M, bias=False)
with torch.no_grad():
  layer.weight.copy_(K)
print(K @ torch.tensor([[0], [1],[0]], dtype=torch.float))
print(layer(torch.tensor([[0,1,0]], dtype=torch.float)))

