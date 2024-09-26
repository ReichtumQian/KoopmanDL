
import torch

class FlowMap(object):
  def __init__(self, dt = 1e-3):
    self._dt = dt

  def step(self, ode, x_init):
    return NotImplementedError
  
  def generate_traj_data(self, ode, n_init, traj_len, x_min, x_max, seed = None):
    if seed is None:
      torch.manual_seed(0)
    else:
      torch.manual_seed(seed)
    x = torch.rand(n_init, ode.dim()) * (x_max - x_min) + x_min
    result = x
    for _ in range(traj_len - 1):
      x = self.step(ode, x)
      torch.cat([result, x], dim = 0)
    return result
  
    

class ForwardEuler(FlowMap):

  def step(self, ode, x_init):
    return x_init + ode.rhs()(x_init) * self._dt

    
    

