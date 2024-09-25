
import torch
import KoopmanDL as kd

data_x = torch.tensor(
  [[1], [2], [3]], dtype=torch.float
)
data_y = torch.tensor(
  [[1.1], [2.1], [3.1]]
)
M = 3
net = kd.TanhFullNet(1, 1, [100, 100, 100])
opt = torch.optim.Adam(net.parameters())
dic = kd.TrainableDictionary(M, net, opt)
solver = kd.EDMDDLSolver(dic, 0.1)
result = solver.solve(data_x, data_y)

