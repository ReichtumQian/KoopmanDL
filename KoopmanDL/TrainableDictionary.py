
import torch
from Dictionary import Dictionary

class TrainableDictionary(Dictionary):

  def __init__(self, M, func, optimizer):
    assert(isinstance(func, torch.nn.Module))
    Dictionary.__init__(self, M, func)
    self.__optimizer = optimizer
    
  def train(self, data_loader, loss_func, K):
    self._func.train()
    for data, labels in data_loader:
      self.__optimizer.zero_grad()
      loss = loss_func(self._func(data), labels)
      loss.backward()
      self.__optimizer.step()


