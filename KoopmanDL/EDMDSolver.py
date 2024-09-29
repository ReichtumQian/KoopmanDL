import torch
from .DataSet import DataSet
import numpy as np
from tqdm import tqdm
from .Device import DEVICE

class EDMDSolver(object):

  def __init__(self, dictionary):
    self._dictionary = dictionary

  def compute_K(self, data_x, data_y):
    PX = self._dictionary(data_x).t()
    PY = self._dictionary(data_y).t()
    A = PY @ PX.t()
    G = PX @ PX.t()
    K =  A @ torch.linalg.pinv(G)
    return K.t()
  
  def compute_eig(self, K):
    eigenvalues, eigenvectors = np.linalg.eig(K.detach().numpy())
    idx = eigenvalues.real.argsort()[::-1]
    self.eigenvalues = torch.from_numpy(eigenvalues[idx])
    self.right_eigenvectors = torch.from_numpy(eigenvectors[:, idx])
    self.left_eigenvectors = torch.from_numpy(np.linalg.inv(eigenvectors))
  
  
  def predict(self, x0, traj_len):
    M = self._dictionary.get_M()
    d = x0.size(1)
    assert(d < M)
    # Compute matrix B
    B = torch.zeros(d, M, dtype=torch.cfloat)
    for i in range(d):
      B[i, i+1] = 1
    # Compute W, V
    W = torch.conj(self.right_eigenvectors)
    V = B @ W
    # Compute \Phi
    def Phi(x):
      Psi = self._dictionary(x)
      phi = torch.conj(self.left_eigenvectors.t()) @ Psi.t().to(torch.cfloat)
      return phi
      
    traj = [x0]
    for _ in range(traj_len - 1):
      x_current = traj[-1]
      phi = Phi(x_current)
      x_next = V @ phi
      x_next = x_next.to(torch.float).t()
      traj.append(x_next)
    return traj
      


class EDMDDLSolver(EDMDSolver):
  
  def __init__(self, dictionary, regularizer):
    super().__init__(dictionary)
    self.__regularizer = regularizer
  
  def compute_K(self, data_x, data_y, reg = None):
    device = data_x.device
    PX = self._dictionary(data_x).t()
    PY = self._dictionary(data_y).t()
    A = PY @ PX.t()
    G = PX @ PX.t()
    if reg is None:
      reg = self.__regularizer
    regularizer = torch.eye(self._dictionary._M) * reg
    regularizer = regularizer.to(device)
    K =  A @ torch.linalg.pinv(G + regularizer)
    return K.t()
  
  def solve(self, data_x, data_y, n_epochs = 100, batch_size = 100):
    device = data_x.device
    data_set = DataSet(data_x, data_y)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    loss_func = torch.nn.MSELoss()
    data_x = data_x.to(DEVICE)
    data_y = data_y.to(DEVICE)
    self._dictionary.get_func().to(DEVICE)
    with tqdm(range(n_epochs), desc="Training") as pbar:
      for epoch in pbar:
        K = self.compute_K(data_x, data_y)
        loss = self._dictionary.train(data_loader, K, loss_func)
        loss_str = f"{loss.item():.2e}"
        pbar.set_postfix(loss=loss_str)
    data_x = data_x.to(device)
    data_y = data_y.to(device)
    self._dictionary.get_func().to(device)