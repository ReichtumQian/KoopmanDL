
import torch
from EDMDSolver import EDMDSolver
from TrainableDictionary import TrainableDictionary
from DataSet import DataSet

# class EDMDDLLossFunc(torch.nn.Module):
#   def __init__(self, dictionary, regularization_factor):
#     super().__init__()
#     self.__dictionary = dictionary
#     self.__regularization_factor = regularization_factor
  
#   def forward(self, outputs, labels, K):
#     mse_loss = torch.F.mse_loss(outputs, labels, reduction='mean')
#     F_norm = torch.norm(K, p='fro')
#     loss = mse_loss + self.__regularization_factor * F_norm
#     return loss



class EDMDDLSolver(EDMDSolver):
  
  def __init__(self, dictionary, regularization_factor):
    # assert(isinstance(dictionary, TrainableDictionary))
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
      loss = loss_func(outputs, labels)
      self._dictionary.train(data_loader, loss_func, K)
      print('epoch: {}, loss: {}'.format(epoch, loss))
      if loss < tolerance:
        print('Converged at epoch: {}, loss: {}'.format(epoch, loss))
        break

  
  



