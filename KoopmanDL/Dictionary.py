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