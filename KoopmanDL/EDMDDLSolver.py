
import torch
from EDMDDLSolver import EDMDSolver
from TrainableDictionary import TrainableDictionary
from DataSet import DataSet

class EDMDDLLossFunc(torch.nn.Module):
  def __init__(self, dictionary, regularization_factor):
    super().__init__()
    self.__dictionary = dictionary
    self.__regularization_factor = regularization_factor
  
  def forward(self, outputs, labels, K):
    mse_loss = torch.F.mse_loss(outputs, labels, reduction='mean')
    F_norm = torch.norm(K, p='fro')
    loss = mse_loss + self.__regularization_factor * F_norm
    return loss



class EDMDDLSolver(EDMDSolver):
  
  def __init__(self, dictionary, regularization_factor):
    assert(isinstance(dictionary, TrainableDictionary))
    super().__init__(dictionary)
    self.__regularization_factor = regularization_factor
  
  def compute_K(self, data_x, data_y):
    G = self._dictionary.compute_G(data_x)
    A = self._dictionary.compute_A(data_x, data_y)
    regularizer = torch.eye(self._dictionary._M) * self.__regularizer_factor
    K = torch.linalg.pinv(G + regularizer) @ A
    return K
  
  def solve(self, data_x, data_y, tolerance = 1e-4, batch_size = 5):
    data_set = DataSet(data_x, data_y)
    K = self.compute_K(data_x, data_y)
    loss_func = EDMDDLLossFunc(self._dictionary, self.__regularization_factor)
    outputs = self._dictionary.get_func()(data_x)
    labels = self._dictionary.get_func()(data_y)
    loss = loss_func(outputs, labels, K)

  
  



