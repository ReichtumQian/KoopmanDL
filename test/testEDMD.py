import torch
import KoopmanDL as kd


func = lambda x: torch.tensor(
  [1, x[0], x[0]**2], dtype=torch.float
)
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
print(dic(data_x))
print(dic(data_y))
print(K)


