import torch
import numpy as np
from scipy.integrate import solve_ivp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer_k(model, x):
  F_dot = model.jacobian(x)
  s1 = x[:, 0:2] - x[:, 2:4]
  s2 = x[:, 4:6] - x[:, 2:4]

  s1 = s1 / torch.norm(s1, dim=1)[:, None]
  s2 = s2 / torch.norm(s2, dim=1)[:, None]

  k1 = torch.einsum('ij, ijk, ik->i', s1, F_dot[:, :, 0:2], s1)
  k2 = torch.einsum('ij, ijk, ik->i', s2, F_dot[:, :, 4:6], s2)

  return ((k1 + k2) / 2).mean()

def getqdot(x, xdot, model, xLeft, xRight):
    N_m = x.shape[0] // 2

    x = np.concatenate((xLeft, x, xRight))
    forces = []
    for i in range(0, N_m):
        triplet = x[np.arange(2*i, 2*i + 6)]
        triplet = triplet[None,:]
        force = model.forward(torch.tensor(triplet).float().to(DEVICE))
        forces.append(force.cpu().detach().numpy())

    forces = np.concatenate(forces).reshape(-1)

    q0 = np.concatenate((x, xdot))
    qdot = np.concatenate((xdot, forces))
    return qdot

def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, 5.0/3.0)

def compute_trajectory(x0, x0dot, model, xLeft, xRight):
    N_m = x0.shape[0] // 2
    q0 = np.concatenate((x0, x0dot))

    t0 = 0
    tf = 40
    Nt = 4001
    sol = solve_ivp(lambda t, q: getqdot(q[:2*N_m], q[2*N_m:], model, xLeft, xRight), [t0,tf], y0=q0, t_eval=np.linspace(t0, tf, Nt))
    y = sol.y
    t = sol.t
    return y, t