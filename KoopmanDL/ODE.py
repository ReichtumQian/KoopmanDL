import torch


class ODE(object):

  def __init__(self, rhs, dim):
    self._dim = dim
    self._rhs = rhs

  def rhs(self):
    return self._rhs

  def dim(self):
    return self._dim


class DuffingOscillator(ODE):

  def __init__(self, alpha, beta, delta):
    dim = 2
    rhs = lambda x: torch.stack([x[:, 1], -delta * x[:, 1] - x[:, 0] * (beta + alpha * x[:, 0]**2)], dim = 1)
    super().__init__(rhs, dim)




