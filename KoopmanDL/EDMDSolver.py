import torch
from .DataSet import DataSet

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


class EDMDDLSolver(EDMDSolver):
  
  def __init__(self, dictionary, regularization_factor):
    super().__init__(dictionary)
    self.__regularization_factor = regularization_factor
  
  def compute_K(self, data_x, data_y):
    PX = self._dictionary.compute_Psi(data_x)
    PY = self._dictionary.compute_Psi(data_y)
    N = data_x.shape[0]
    A = PY @ PX.t() / N
    G = PX @ PX.t() / N
    regularizer = torch.eye(self._dictionary._M) * self.__regularization_factor
    K = A @ torch.linalg.pinv(G + regularizer) 
    return K
  
  def solve(self, data_x, data_y, tolerance = 1e-10, num_epochs = 100, batch_size = 100):
    data_set = DataSet(data_x, data_y)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    loss_func = torch.nn.MSELoss()
    for epoch in range(num_epochs):
      K = self.compute_K(data_x, data_y)
      outputs = self._dictionary.get_func()(data_x)
      labels = self._dictionary.get_func()(data_y)
      # Compute K Psix
      outputs_unsqueezed = outputs.unsqueeze(2)
      K_expanded = K.unsqueeze(0).expand(outputs.size(0), -1, -1)
      outputs = torch.bmm(K_expanded, outputs_unsqueezed).squeeze(2)

      loss = loss_func(outputs, labels)
      self._dictionary.train(data_loader, loss_func)
      print('epoch: {}, loss: {}'.format(epoch, loss))
      if loss < tolerance:
        print('Converged at epoch: {}, loss: {}'.format(epoch, loss))
        break