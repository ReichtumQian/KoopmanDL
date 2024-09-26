
import torch

class FlowMap(object):
  def __init__(self, dt = 1e-3):
    self._dt = dt

  def step(self, ode, x_init):
    return NotImplementedError
  
  def generate_traj_data(self, ode, n_init, traj_len, traj_t_step, x_min, x_max, seed = None):
    if seed is None:
      torch.manual_seed(0)
    else:
      torch.manual_seed(seed)
    n_step = int(traj_t_step / self._dt)
    x = torch.rand(n_init, ode.dim()) * (x_max - x_min) + x_min
    result = x
    for _ in range(traj_len - 1):
      for __ in range(n_step):
        x = self.step(ode, x)
      result = torch.cat([result, x], dim = 0)
    return result
  
  def generate_next_data(self, ode, x, traj_t_step):
    n_step = int(traj_t_step / self._dt)
    result = x
    for __ in range(n_step):
      result = self.step(ode, result)
    return result

    
  
    

class ForwardEuler(FlowMap):

  def step(self, ode, x_init):
    return x_init + ode.rhs()(x_init) * self._dt

    
    

