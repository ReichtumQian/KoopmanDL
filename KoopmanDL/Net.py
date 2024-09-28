import torch


class TanhResBlock(torch.nn.Module):

  def __init__(self, n_input, n_output):
    super().__init__()
    self.__linear1 = torch.nn.Linear(n_input, n_output)
    self.__tanh = torch.nn.Tanh()
    self.__linear2 = torch.nn.Linear(n_output, n_output)

    if n_input != n_output:
      self.__shortcut = torch.nn.Linear(n_input, n_output)
    else:
      self.__shortcut = torch.nn.Identity()

  def forward(self, x):
    identity = self.__shortcut(x)
    out = self.__linear1(x)
    out = self.__tanh(out)
    out = self.__linear2(out)
    out += identity
    out = self.__tanh(out)
    return out


class TanhResNet(torch.nn.Module):

  def __init__(self, n_input, n_output, hidden_layer_sizes):
    super().__init__()
    layers = []
    input_size = n_input
    for hidden_size in hidden_layer_sizes:
      layers.append(TanhResBlock(input_size, hidden_size))
      input_size = hidden_size
    layers.append(torch.nn.Linear(hidden_layer_sizes[-1], n_output))
    self.__network = torch.nn.Sequential(*layers)

  def forward(self, x):
    return self.__network(x)


class TanhResNetWithNonTrainable(TanhResNet):

  def __init__(self, n_input, n_output, hidden_layer_sizes, n_nontrainable):
    super().__init__(n_input, n_output - n_nontrainable, hidden_layer_sizes)

  def forward(self, x):
    net_output = super().forward(x)
    result = torch.cat([torch.ones(x.size(0), 1),
                        x.detach(), net_output],
                       dim=1)
    return result
