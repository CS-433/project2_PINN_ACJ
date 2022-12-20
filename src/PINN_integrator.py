import numpy as np
import torch
from scipy.integrate import solve_ivp

def getqdot(x, xdot, model, xLeft, xRight):
    N_m = x.shape[0] // 2

    x = np.concatenate((xLeft, x, xRight))
    forces = []
    for i in range(0, N_m):
        triplet = x[np.arange(2*i, 2*i + 6)]
        triplet = triplet[None,:]
        force = model.forward(torch.tensor(triplet).float())
        forces.append(force.detach().numpy())

    forces = np.concatenate(forces).reshape(-1)

    q0 = np.concatenate((x, xdot))
    qdot = np.concatenate((xdot, forces))
    return qdot

def compute_trajectory(x0, x0dot, model, xLeft, xRight):
    N_m = x0.shape[0] // 2
    q0 = np.concatenate((x0, x0dot))

    t0 = 0
    tf = 30
    Nt = 3000
    sol = solve_ivp(lambda t, q: getqdot(q[:2*N_m], q[2*N_m:], model, xLeft, xRight), [t0,tf], y0=q0, t_eval=np.linspace(t0, tf, Nt))
    y = sol.y
    t = sol.t
    return y, t