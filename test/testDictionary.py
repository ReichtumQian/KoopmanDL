import KoopmanDL
import torch

func = lambda x: torch.tensor(
    [torch.sin(x[0]),
     torch.cos(x[1])])
data_x = torch.tensor([[1, 1], [2, 4]])
data_y = torch.tensor([[2, 4], [3, 9]])
print(func(torch.tensor([1, 2])))
Dic = KoopmanDL.Dictionary(2, func)
Psi = Dic(data_x)
G = Dic.compute_G(data_x)
print(G)
