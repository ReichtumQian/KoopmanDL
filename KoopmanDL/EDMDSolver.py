import torch

class EDMDSolver(object):

  def __init__(self, dictionary):
    self._dictionary = dictionary

  def compute_K(self, data_x, data_y):
    PX = self._dictionary.compute_Psi(data_x)
    PY = self._dictionary.compute_Psi(data_y)
    # K = PY @ torch.linalg.pinv(PX)
    N = data_x.shape[0]
    A = PY @ PX.t() / N
    G = PX @ PX.t() / N
    K = A @ torch.linalg.pinv(G)
    return K

