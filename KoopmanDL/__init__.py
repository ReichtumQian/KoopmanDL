
import torch
import os

from .DataSet import DataSet
from .Dictionary import Dictionary, TrainableDictionary, RBFDictionary
from .EDMDSolver import EDMDSolver, EDMDDLSolver
from .FlowMap import ForwardEuler, FlowMap
from .ODE import DuffingOscillator, VanDerPolOscillator
from .Net import TanhResNet, TanhResNetWithNonTrainable

# check if GPU is available
if torch.cuda.is_available():
  print("CUDA is available. GPU is being used.")
else:
  print("CUDA is not available. Using CPU for computation.")

# set the project root dir
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Project root dir is set to: " + root_dir)
