
import torch
from .DataSet import DataSet
from .Dictionary import Dictionary, TrainableDictionary, RBFDictionary
from .EDMDSolver import EDMDSolver, EDMDDLSolver
from .FlowMap import ForwardEuler, FlowMap
from .ODE import DuffingOscillator
from .Net import TanhResNet, TanhResNetWithNonTrainable

if torch.cuda.is_available():
  print("CUDA is available. GPU is being used.")
else:
  print("CUDA is not available. Using CPU for computation.")


