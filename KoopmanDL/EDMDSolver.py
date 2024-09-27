import torch
from .DataSet import DataSet
import sys

class EDMDSolver(object):

  def __init__(self, dictionary):
    self._dictionary = dictionary

  def compute_K(self, data_x, data_y):
    PX = self._dictionary.compute_Psi(data_x)
    PY = self._dictionary.compute_Psi(data_y)
    # K = PY @ torch.linalg.pinv(PX)
    N = data_x.shape[0]
    #A = PY @ PX.t() / N
    #G = PX @ PX.t() / N
    #K = A @ torch.linalg.pinv(G)
    A = PX @ PY.t()
    G = PX @ PX.t()
    K = torch.linalg.pinv(G + regularizer) @ A 
    return K


class EDMDDLSolver(EDMDSolver):
  
  def __init__(self, dictionary, regularization_factor):
    super().__init__(dictionary)
    self.__regularization_factor = regularization_factor
  
  def compute_K(self, data_x, data_y):
    PX = self._dictionary.compute_Psi(data_x)
    PY = self._dictionary.compute_Psi(data_y)
    N = data_x.size(0)
    #A = PY @ PX.t() / N
    #G = PX @ PX.t() / N
    A = PX @ PY.t()
    G = PX @ PX.t()
    regularizer = torch.eye(self._dictionary._M) * self.__regularization_factor
    K = torch.linalg.pinv(G + regularizer) @ A 
    return K
  
  def solve(self, data_x, data_y, tolerance = 1e-10, num_epochs = 100, batch_size = 100):
    data_set = DataSet(data_x, data_y)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    loss_func = torch.nn.MSELoss()
    for epoch in range(num_epochs):
      K = self.compute_K(data_x, data_y).detach()
      #print(K)
      #sys.stdout.flush()
      #self._dictionary.get_func().set_output_layer(K)
      outputs = self._dictionary.get_func()(data_x) @ K
      labels = self._dictionary.get_func()(data_y)
      #print(labels)
      #print(outputs)
      #print(epoch)
      #print("hello")
      loss = loss_func(outputs, labels)
      self._dictionary.train(data_loader, K, loss_func)
      print('epoch: {}, loss: {}'.format(epoch+1, loss))
      if loss < tolerance:
        print('Converged at epoch: {}, loss: {}'.format(epoch, loss))
        break