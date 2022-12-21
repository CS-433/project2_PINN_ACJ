import torch
from torch.autograd import grad
from functorch import vmap, vjp
from functorch import jacrev, jacfwd
from torch import nn

class NNApproximator(nn.Module):
  '''
  '''
  def __init__(self, dim_input=6, dim_output=2, num_hidden=2, dim_hidden=1, activation=nn.Tanh()):
    super().__init__()

    # self.layer_in = nn.Linear(dim_input, dim_hidden)
    self.layer_in = nn.Linear(4, dim_hidden)
    self.layer_out = nn.Linear(dim_hidden, dim_output)
    self.k = nn.Parameter(torch.tensor(50.0, requires_grad=False))

    num_middle = num_hidden - 1
    self.middle_layers = nn.ModuleList(
        [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
    )
    self.activation = activation
    self.scale_x = 10.0
    self.scale_f = 30.0
    self.scale_grad_f = self.scale_f / self.scale_x

  def set_X_scale(scale_x):
    self.scale_x = scale_x
    self.scale_grad_f = self.scale_f / self.scale_x

  def set_F_scale(scale_f):
    self.scale_f = scale_f
    self.scale_grad_f = self.scale_f / self.scale_x

  # for newer PINN loss
  def forward(self, x):
    x_unit = x.view(-1,6) / self.scale_x
    s = torch.hstack((x_unit[:, 0:2] - x_unit[:, 2:4], x_unit[:, 4:6] - x_unit[:, 2:4]))
    out = self.activation(self.layer_in(s))
    for layer in self.middle_layers:
      out = self.activation(layer(out))
    return self.layer_out(out) * self.scale_f

  def _get_force_truth(self, x):
    k = 50.0
    g = torch.tensor([[0.0, -9.81]])
    L0 = 7.0
    x1 = x.view(-1,6)[:,:2]
    x2 = x.view(-1,6)[:,2:4]
    x3 = x.view(-1,6)[:,4:]
    dx1 = x2 - x1
    dx2 = x3 - x2
    dx1_norm = torch.sqrt(torch.sum(dx1 ** 2, dim=1))[:,None]
    dx2_norm = torch.sqrt(torch.sum(dx2 ** 2, dim=1))[:,None]
    f1 = -k * (dx1_norm - L0) * (dx1 / dx1_norm)
    f2 = k * (dx2_norm - L0) * (dx2 / dx2_norm)
    return g + f1 + f2

  # reference for implementing derivatives for batched inputs
  # https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html
  def jacobian(self, x):
    jac = vmap(jacrev(self.forward))
    return jac(x).squeeze()