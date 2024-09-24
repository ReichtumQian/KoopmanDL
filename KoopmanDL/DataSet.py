import torch

class DataSet(torch.utils.data.Dataset):
  
  def __init__(self, data, labels):
    self.__data = data
    self.__labels = labels

  def __len__(self):
    return len(self.__data)

  def __getitem__(self, idx):
    return self.__data[idx], self.__labels[idx]


