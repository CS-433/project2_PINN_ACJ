import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse

class SpringMassSystem():
    def __init__(self, ndim, N_m, L0, k0, m0):
        self.ndim = ndim
        self.N_m = N_m
        N_s = N_m + 1
        self.k_list = np.ones(N_s) * k0
        self.L_list = np.ones(N_s) * L0
        self.m_list = np.ones(N_m) * m0
        self.gravity = 9.81

        # inverse mass matrix
        self.Minv = sparse.diags(
            np.kron(1 / self.m_list, np.ones(ndim)), 
            format='csr')

        self.c_list = None      # stress coefficients
        self.K = None           # stiffness matrix

        self.xL = np.zeros((1,ndim))
        self.xR = self.xL.copy()
        self.xR[0,0] = N_s * L0

        self.X = np.zeros((N_m, ndim))
        self.X[:,0] = self.xL[0,0] + (1 + np.arange(N_m)) * L0


    def get_initial_X_aug(self):
        X = self.X.copy()
        Xdot = np.zeros_like(X)
        X_aug = np.vstack((
            np.reshape(X,(-1,1)), 
            np.reshape(Xdot, (-1,1))
            ))
        return X_aug.ravel()

    def initialize_as_caternary(self, extension_ctr=1e-2):
        k0 = np.mean(self.k_list)
        m0 = np.mean(self.m_list)
        L0 = np.mean(self.L_list)
        L_tot = np.sum(self.L_list)

        s = np.linspace(-L_tot/2, L_tot/2, self.N_m + 2)
        T_ctr = k0 * extension_ctr * (s[1] - s[0])
        a = T_ctr / (self.gravity * m0/L0)
        xs = a * np.arcsinh(s / a)
        ys = np.sqrt(s**2 + a**2)

        xs_m = xs[1:-1]
        ys_m = ys[1:-1]
        extended_lengths = np.sqrt((xs[1:] - xs[:-1])**2 + (ys[1:] - ys[:-1])**2)

        ys_spring = np.sqrt((0.5*(s[1:] + s[:-1]))**2 + a**2)
        Ts = np.abs(T_ctr*ys_spring / a)
        rest_lengths = extended_lengths - Ts / k0

        self.L_list = rest_lengths
        self.X = np.c_[xs_m, ys_m]
        self.xL = np.array([[xs[0], ys[0]]])
        self.xR = np.array([[xs[-1], ys[-1]]])





    def update_stiffness_coeffs(self):
        Xt = np.vstack((self.xL,self.X,self.xR))
        strain_list = 1. - self.L_list / la.norm(Xt[1:,:] - Xt[:-1,:], axis=1)
        self.c_list = self.k_list * strain_list

    def build_stiffness_matrix(self):
        self.K = sparse.diags(
            diagonals=[
                np.kron( self.c_list[1:-1], np.ones(self.ndim)), 
                np.kron(-self.c_list[:-1]-self.c_list[1:], np.ones(self.ndim)),
                np.kron( self.c_list[1:-1], np.ones(self.ndim))
            ], 
            offsets=[-self.ndim, 0, self.ndim], 
            format='csr',
        )


    def build_bc(self):
        f_bc = np.zeros((self.N_m * self.ndim,1))
        f_bc[:self.ndim, 0] = self.c_list[0] * self.xL.ravel()
        f_bc[(self.N_m - 1)*self.ndim:, 0] = self.c_list[-1] * self.xR.ravel()
        return f_bc
    
    def build_ext_f(self):
        if self.ndim > 1:
            g = np.array([[0],[-self.gravity]])
            f = np.kron(np.ones((self.N_m,1)), g)
        else:
            f = np.kron(np.ones((self.N_m,1)), np.zeros(1))
            
        return f

    def compute_Xddot(self, X):
        self.X = X
        self.update_stiffness_coeffs()
        self.build_stiffness_matrix()
        f_bc = self.build_bc()
        f_g = self.build_ext_f()
        X_ddot = self.Minv @ self.K @ np.reshape(self.X, (-1,1)) + self.Minv @ f_bc + f_g
        X_ddot = np.reshape(X_ddot, (-1,self.ndim))
        return X_ddot

    def compute_augmented_xdot(self, t, X_aug):
        # X_aug has shape [2 * N_m * ndim, 1]

        X = np.reshape(X_aug[:(self.N_m * self.ndim)], (-1, self.ndim))
        Xddot = self.compute_Xddot(X)
        X_aug_dot = np.concatenate((
                X_aug[(self.N_m * self.ndim):],           # Xdot
                Xddot.ravel()
                )
            )
        return X_aug_dot