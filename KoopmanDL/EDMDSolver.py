import torch

class EDMDSolver(object):

  def __init__(self, dictionary):
    self._dictionary_ = dictionary

  def compute_K(self, data_x, data_y):
    G = self._dictionary_.compute_G(data_x)
    A = self._dictionary_.compute_A(data_x, data_y)
    PG = torch.linalg.pinv(G)
    K = PG @ A
    return K

