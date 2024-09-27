import torch
from .DataSet import DataSet
import numpy as np
from tqdm import tqdm

class EDMDSolver(object):

  def __init__(self, dictionary):
    self._dictionary = dictionary

  def compute_K(self, data_x, data_y):
    PX = self._dictionary(data_x).t()
    PY = self._dictionary(data_y).t()
    N = data_x.shape[0]
    A = PY @ PX.t() / N
    G = PX @ PX.t() / N
    K = A @ torch.linalg.pinv(G)
    return K
  
  def eig_decomp(self, K):
    self.eigenvalues, self.eigenvectors = np.linalg.eig(K.detach().numpy())
    idx = self.eigenvalues.real.argsort()[::-1]
    self.eigenvalues = self.eigenvalues[idx]
    self.eigenvectors = self.eigenvectors[:, idx]
    self.eigenvectors_inv = np.linalg.inv(self.eigenvectors)

  def eigenfunctions(self, data_x):
    psi_x = self._dictionary(data_x)
    val = np.matmul(psi_x, self.eigenvectors)
    return val


class EDMDDLSolver(EDMDSolver):
  
  def __init__(self, dictionary, regularizer):
    super().__init__(dictionary)
    self.__regularizer = regularizer
  
  def compute_K(self, data_x, data_y, reg = None):
    PX = self._dictionary(data_x).t()
    PY = self._dictionary(data_y).t()
    N = data_x.size(0)
    A = PY @ PX.t()
    G = PX @ PX.t()
    if reg is None:
      reg = self.__regularizer
    regularizer = torch.eye(self._dictionary._M) * reg
    K = A @ torch.linalg.pinv(G + regularizer) 
    return K
  
  def solve(self, data_x, data_y, n_epochs = 100, batch_size = 100):
    data_set = DataSet(data_x, data_y)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    loss_func = torch.nn.MSELoss()
    with tqdm(range(n_epochs), desc="Training") as pbar:
      for epoch in pbar:
        K = self.compute_K(data_x, data_y)
        self._dictionary.get_func().set_output_layer(K)
        loss = self._dictionary.train(data_loader, loss_func)
        loss_str = f"{loss.item():.2e}"
        pbar.set_postfix(loss=loss_str)