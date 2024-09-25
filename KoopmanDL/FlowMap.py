
import torch

class FlowMap(object):
  def __init__(self, dt = 1e-3):
    self._dt = dt

  def step(self, ODE, x_init):
    return NotImplementedError

class ForwardEuler(FlowMap):

  def step(self, ODE, x_init, t):
    return x_init + ODE.rhs()(x_init, t) * self._dt

    
    

