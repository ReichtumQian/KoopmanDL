import torch


class TanhNet(torch.nn.Module):

  def __init__(self, n_input, n_output, hidden_layer_sizes):
    super().__init__()
    layers = []
    for i in range(len(hidden_layer_sizes)):
      layers.append(
          torch.nn.Linear(n_input if i == 0 else hidden_layer_sizes[i - 1],
                          hidden_layer_sizes[i]))
      layers.append(torch.nn.Tanh())
    layers.append(torch.nn.Linear(hidden_layer_sizes[-1], n_output))
    self.__network = torch.nn.Sequential(*layers)

  def forward(self, x):
    return self.__network(x)


class TanhNetWithNonTrainable(TanhNet):

  def __init__(self, n_input, n_output, hidden_layer_sizes, n_nontrainable):
    super().__init__(n_input, n_output - n_nontrainable, hidden_layer_sizes)

  def forward(self, x):
    net_output = super().forward(x)
    if x.dim() is 1:
      result = torch.cat([torch.ones(1), x, net_output], dim=0)
    else:
      result = torch.cat([torch.ones(x.size(0), 1), x, net_output], dim=1)
    return result
