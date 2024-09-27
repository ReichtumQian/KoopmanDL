
import torch
import KoopmanDL as kd

data_x = torch.tensor(
  [[1], [2], [3], [4], [5]], dtype=torch.float
)
data_y = torch.tensor(
  [[2], [3], [4], [5], [6]], dtype=torch.float
)
M = 3
net = kd.TanhNetWithNonTrainable(1, M, [10, 10], 2)
opt = torch.optim.Adam(net.parameters())
dic = kd.TrainableDictionary(M, net, opt)
solver = kd.EDMDDLSolver(dic, 0.1)
solver.solve(data_x, data_y)
K = solver.compute_K(data_x, data_y)
print(K)
