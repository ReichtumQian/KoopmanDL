
import torch
from Dictionary import Dictionary

class TrainableDictionary(Dictionary):

  def __init__(self, M, func, optimizer):
    assert(isinstance(func, torch.nn.Module))
    Dictionary.__init__(self, M, func)
    self.__optimizer = optimizer
    
  def train(self, data_loader, loss_func, K):
    epoch_loss = 0
    num_batches = 0
    self._func.train()
    for data, labels in data_loader:
      self.__optimizer.zero_grad()
      loss = loss_func(self._func(data), labels, K)
      loss.backward()
      self.__optimizer.step()

      epoch_loss += loss.item()
      num_batches += 1
    average_loss = epoch_loss / num_batches
    return average_loss
    

