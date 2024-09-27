import torch

class DataSet(torch.utils.data.Dataset):
  
  def __init__(self, data, labels):
    self.__data = data
    self.__labels = labels

  def __len__(self):
    return len(self.__data)

  def __getitem__(self, idx):
    return self.__data[idx], self.__labels[idx]
  
  def data(self):
    return self.__data
  
  def labels(self):
    return self.__labels

  def split(self, train_size):
    torch_train, torch_valid = torch.utils.data.random_split(self, [train_size, len(self) - train_size])
    train = SubSet(self, torch_train.indices)
    valid = SubSet(self, torch_valid.indices)
    return train, valid


class SubSet(torch.utils.data.Subset):
  
  def __init__(self, dataset, indices):
    super(SubSet, self).__init__(dataset, indices)
  
  def data(self):
    return torch.stack([self.dataset.data()[i] for i in self.indices], dim = 0)
  
  def labels(self):
    return torch.stack([self.dataset.labels()[i] for i in self.indices], dim = 0)
