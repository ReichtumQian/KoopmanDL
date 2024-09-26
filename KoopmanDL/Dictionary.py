import torch


class Dictionary(object):

  def __init__(self, M, func):
    self._func = func
    self._M = M

  def compute_Psi(self, data_x):
    N = data_x.size(dim = 0)
    M = self._M
    psi = torch.zeros([M, N])
    for i in range(N):
      psi[:, i] = self._func(data_x[i])
    return psi
  def get_func(self):
    return self._func




class TrainableDictionary(Dictionary):

  def __init__(self, M, func, optimizer):
    assert(isinstance(func, torch.nn.Module))
    Dictionary.__init__(self, M, func)
    self.__optimizer = optimizer
    
  def train(self, data_loader, loss_func, n_epochs = 2):
    for _ in range(n_epochs):
      self._func.train()
      for data, labels in data_loader:
        self.__optimizer.zero_grad()
        loss = loss_func(self._func(data), self._func(labels))
        loss.backward()
        self.__optimizer.step()