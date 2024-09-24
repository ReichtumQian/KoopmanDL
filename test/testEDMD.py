import torch
import KoopmanDL as kd


func = lambda x: torch.tensor(
  [1, x[0], x[0]**2]
)
data_x = torch.tensor(
  [[1], [2], [3]]
)
data_y = torch.tensor(
  [[1.1], [2.1], [3.1]]
)
M = 3
dic = kd.Dictionary(M, func)
print(dic.compute_Psi(data_x))
print(dic.compute_G(data_x))
print(dic.compute_A(data_x, data_y))
solver = kd.EDMDSolver(dic)
print(solver.compute_K(data_x, data_y))


