import torch


class Dictionary(object):

  def __init__(self, M, func):
    """Initialize an dictionary.

    Args:
        M (int) : the output dim of func
        func (Any type): the dictionary functions R^d -> R^M, where d is the dim of data, M is the number of dictionary functions.
    """
    self._func = func
    self._M = M

  def compute_Psi(self, data_x):
    """Compute Psi(data_x)

    Args:
        data_x (torch.tensor): R^{N*d}, where N is the number of data, d is the dim of data.

    Returns:
        torch.tensor : Matrix Psi of the form R^{N*M}
    """
    N = data_x.size(dim = 0)
    M = self._M
    psi = torch.zeros([N, M])
    for i in range(N):
      psi[i] = self._func(data_x[i])
    return psi

  def compute_G(self, data_x):
    """Compute G = 1/N sum Psi(x)^T Psi(x)

    Args:
        data_x (torch.tensor): R^{N*d}, where N is the number of data, d is the dim of data.
    
    Returns:
        torch.tensor : Matrix G of the form R^{M*M}
    """
    PsiX = self.compute_Psi(data_x)
    N = data_x.size(dim = 0)
    M = self._M
    G = torch.zeros([M, M])
    for i in range(N):
      G += PsiX[i].unsqueeze(1) @ PsiX[i].unsqueeze(0)
    G = G / N
    return G

  def compute_A(self, data_x, data_y):
    PsiX = self.compute_Psi(data_x)
    PsiY = self.compute_Psi(data_y)
    N = data_x.size(dim = 0)
    M = self._M
    A = torch.zeros([M, M])
    for i in range(N):
      A += PsiX[i].unsqueeze(1) @ PsiY[i].unsqueeze(0)
    A = A / N
    return A

  def get_func(self):
    return self._func