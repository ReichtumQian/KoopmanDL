import scipy.cluster
import torch
import scipy
import numpy as np


class Dictionary(object):

  def __init__(self, M, func):
    self._func = func
    self._M = M
  def get_func(self):
    return self._func
  
  def __call__(self, x):
    return self._func(x)


class TrainableDictionary(Dictionary):

  def __init__(self, M, func, optimizer):
    assert(isinstance(func, torch.nn.Module))
    Dictionary.__init__(self, M, func)
    self.__optimizer = optimizer
    
  def train(self, data_loader, K, loss_func, n_epochs = 2):
    for _ in range(n_epochs):
      self._func.train()
      for data, labels in data_loader:
        self.__optimizer.zero_grad()
        K = K.detach()
        X = self(data) @ K
        Y = self(labels)
        loss = loss_func(X, Y)
        loss.backward()
        self.__optimizer.step()
    return loss

class RBFDictionary(Dictionary):

  def __init__(self, M = 100, regularization_factor = 1e-4):
    super().__init__(M, None)
    self.__regularization_factor = regularization_factor
  
  def build(self, data):
    self.__centers = scipy.cluster.vq.kmeans(data, self._M)[0]
    def func(x):
      rbfs = []
      for n in range(self._M):
        r = scipy.spatial.distance.cdist(x, np.matrix(self.__centers[n, :]))
        rbf = scipy.special.xlogy(r**2, r + self.__regularization_factor)
        rbfs.append(rbf)
      
      rbfs = np.array(rbfs).squeeze()
      rbfs = torch.tensor(rbfs, dtype=torch.float64)
      rbfs = rbfs.transpose(0, 1).reshape(x.shape[0], -1)
      
      ones = torch.ones((rbfs.shape[0], 1), dtype=torch.float64)
      results = torch.cat([ones, x, rbfs], dim=1)
      return results
  
    self._func = func
  