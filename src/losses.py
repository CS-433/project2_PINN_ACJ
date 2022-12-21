import torch
from functorch import vmap
import numpy as np

def compute_data_loss_force(model, x_tr, y_tr):
  """
  Computes the data regression loss for a model, using the training inputs `x_tr` and labels `y_tr`.
  """
  return 0.5 * torch.mean((model.forward(x_tr).flatten() - y_tr.flatten()) ** 2) / (model.scale_f ** 2)

def compute_PINN_loss(model, x):
  """
  Computes the PINN loss for a PINN model at the collocation points `x`.
  """
  F_dot = model.jacobian(x)
  s1 = x[:, 0:2] - x[:, 2:4]
  s2 = x[:, 4:6] - x[:, 2:4]

  s1 = s1 / torch.norm(s1, dim=1)[:, None]
  s2 = s2 / torch.norm(s2, dim=1)[:, None]

  s1rot = s1 @ torch.from_numpy(np.array([[0, -1], [1, 0]]).T).float()
  s2rot = s2 @ torch.from_numpy(np.array([[0, -1], [1, 0]]).T).float()

  # spring force direction constraints
  f1_ax = torch.einsum('ij, ijk, ik->i', s1, F_dot[:, :, 0:2], s1) - model.k
  f2_ax = torch.einsum('ij, ijk, ik->i', s2, F_dot[:, :, 4:6], s2) - model.k
  f1_perp = torch.einsum('ij, ijk, ik->i', s1, F_dot[:, :, 0:2], s1rot)
  f2_perp = torch.einsum('ij, ijk, ik->i', s2, F_dot[:, :, 4:6], s2rot)

  # conservative force constraints
  symmetrize = lambda A: A[0,1] - A[1,0]
  f_cons2 = vmap(symmetrize)(F_dot[:,:,2:4])
  f_cons1 = vmap(symmetrize)(F_dot[:,:,0:2])
  f_cons3 = vmap(symmetrize)(F_dot[:,:,4:6])

  loss = (f1_ax ** 2).mean() + (f2_ax ** 2).mean()
  loss += (f1_perp ** 2).mean() + (f2_perp ** 2).mean()
  loss += (f_cons1 ** 2).mean() + (f_cons2 ** 2).mean() + (f_cons3 ** 2).mean()
  loss /= model.scale_grad_f ** 2
  return loss


# def compute_PINN_loss(model, x):
#   F = model.forward(x)

#   s1 = x[:, 0:2] - x[:, 2:4]
#   s2 = x[:, 4:6] - x[:, 2:4]

#   s1_unit = s1 / torch.norm(s1, dim=1)[:, None]
#   s2_unit = s2 / torch.norm(s2, dim=1)[:, None]

#   s1rot = s1_unit @ torch.from_numpy(np.array([[0, -1], [1, 0]]).T).float()
#   s2rot = s2_unit @ torch.from_numpy(np.array([[0, -1], [1, 0]]).T).float()

#   vecjac_s1 = grad(outputs=F, inputs=x, grad_outputs=s1_unit, retain_graph=True)[0]
#   f1_ax = (torch.sum(s1_unit * vecjac_s1[:,0:2], dim=1) - model.k) #/ k
#   f1_perp = torch.sum(s1rot * vecjac_s1[:,0:2], dim=1)

#   vecjac_s2 = grad(outputs=F, inputs=x, grad_outputs=s2_unit, retain_graph=True)[0]
#   f2_ax = (torch.sum(s2_unit * vecjac_s2[:,4:6], dim=1) - model.k) #/ k
#   f2_perp = torch.sum(s2rot * vecjac_s2[:,4:6], dim=1)

#   # extract the jacobian entry [0,3]
#   m01 = torch.zeros_like(s1)
#   m01[:, 0] = 1
#   vecjac_01 = grad(outputs=F, inputs=x, grad_outputs=m01, retain_graph=True)[0][:,3] #[:, [1,3,5]]

#   # extract the jacobian entry [1,2]
#   m10 = m01[:, [1, 0]]
#   vecjac_10 = grad(outputs=F, inputs=x, grad_outputs=m10)[0][:, 2]#[:, [0,2,4]]
#   # conservative force constraint: F should be curl-free
#   f_cons = vecjac_01 - vecjac_10

#   loss = 0
#   loss += (f1_perp ** 2).mean() + (f2_perp ** 2).mean()
#   loss += (f1_ax ** 2).mean() + (f2_ax ** 2).mean()
#   loss += (f_cons ** 2).mean()

#   # re-scaling
#   loss /= model.scale_grad_f ** 2