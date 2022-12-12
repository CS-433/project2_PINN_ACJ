import numpy as np
import torch
from scipy.integrate import solve_ivp

def getqdot_RNN(x, xdot, model, xLeft, xRight, x_prev, fs_prev):
    N_m = x.shape[0] // 2

    # build the full position vector
    x = np.concatenate((xLeft, x, xRight))
    x_prev = np.concatenate((xLeft, x_prev, xRight))
    forces = []

    for i in range(0, N_m):
        # select positions & forces for the i-th mass
        indices = np.arange(2*i, 2*i + 6)
        x_triplet = x[indices]
        x_prev_triplet = x_prev[indices]
        f_prev = fs_prev[i, :]   # shape: (1 x 2)
        # TODO: combine the x, x_prev, f_prev values into the correct layout for input to the NN
        inputs = None
        # inputs = np.concatenate((x_triplet.ravel(), x_prev_triplet.ravel(), f_prev.ravel()))
        inputs = torch.from_numpy(inputs).float()
        force = model.forward(inputs)
        forces.append(force.detach().numpy())

    forces = np.vstack(forces)

    qdot = np.concatenate((xdot, forces.ravel()))
    return qdot, forces

def compute_trajectory_RNN(x0, x0dot, model, xLeft, xRight):
    N_m = x0.shape[0] // 2
    q0 = np.concatenate((x0, x0dot))

    t0 = 0
    tf = 20
    # when using explicit euler, we *must* use a tiny timestep
    Nt = 4001                         
    ts = np.linspace(t0, tf, Nt)
    dt = ts[1] - ts[0]
    q = q0
    qs = [q]
    x_prev = q0[:2*N_m]             # TODO: choose appropriate initial value for x_prev
    f_prev = np.zeros((N_m,2))      # TODO: choose appropriate initial value for f_prev

    for t in ts:
      qdot, forces = getqdot(q[:2*N_m], q[2*N_m:], model, xLeft, xRight, x_prev, f_prev)
      q += dt * qdot
      qs.append(q.copy())
      x_prev = q[:2*N_m]
      f_prev = forces

    qs = np.c_[qs]
    return qs, ts


# # # example usage

# N_m = 5
# xLeft = x[0, 13:15]
# xRight = x[0, 15:17]

# # # the vector 'x0' contains the initial positions of the *movable* masses
# # # i.e. x0.shape = [N_m * 2]
# x0 = x[0, [1, 2, 5, 6, 9, 10]]
# x0dot = np.zeros_like(x0)

# model = NNApproximator(dim_input=6+6+2, dim_output=2)

# y, t = compute_trajectory_RNN(x0, x0dot, model, xLeft, xRight)