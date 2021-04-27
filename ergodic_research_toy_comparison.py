# Optimal communication between neighbors
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import random
import time

import casadi
import numpy as np 
import torch
from scipy import integrate
import pandas as pd
import pickle as pkl

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# %%
def get_h_k(new_k, U_shape):
    U_bounds = [[0, U_bound] for U_bound in U_shape]
    integrand = lambda *x : np.prod(np.cos(np.array(x)*np.array(new_k))**2)
    integral_result, _ = integrate.nquad(integrand, U_bounds)
    return np.sqrt(integral_result)

def get_fourier_fn_k(new_k, U_shape):
    # only torch one because we need gradient
    h_k = torch.tensor(get_h_k(new_k, U_shape))
    return lambda x : (1/h_k) * torch.prod(torch.cos(x*torch.tensor(new_k)))

def casadi_prod(x, n):
    result = 1
    for i in range(n):
        result *= x[i] 
    return result

def get_casadi_fourier_fn_k(new_k, U_shape):
    h_k = get_h_k(new_k, U_shape)
    return lambda x : (1/h_k) * casadi_prod(casadi.cos(x*casadi.MX(new_k)), len(U_shape))

def get_lambda_k(new_k, n):
    s = (n+1)/2
    lambda_k = 1 / (1 + np.linalg.norm(new_k)**2)**s
    return lambda_k

def get_mu_k(mu_dist, f_k, U_shape):
    U_bounds = [[0, U_bound] for U_bound in U_shape]
    # is mu defined everywhere in the bounds
    integrand = lambda *x: mu_dist(np.array(x)) * f_k(torch.tensor(x)).numpy()
    integral_result, _ = integrate.nquad(integrand, U_bounds)
    return integral_result

def get_grad_fourier_fn_k(fourier_k):
    return lambda x: torch.autograd.functional.jacobian(fourier_k, x)
    
def get_functions_k(k, mu_dist, U_shape, N):
    n = len(U_shape)
    new_k = np.array(k)*np.pi/np.array(U_shape)
    fourier_k = get_fourier_fn_k(new_k, U_shape)
    grad_fourier_k = get_grad_fourier_fn_k(fourier_k)
    lambda_k = get_lambda_k(new_k, n)
    mu_k = get_mu_k(mu_dist, fourier_k, U_shape)
    return {'fourier_k': fourier_k, 
            'grad_fourier_k': grad_fourier_k, 
            'mu_k': mu_k, 
            'C_k': np.zeros(N, 1),
            'lambda_k': lambda_k}

def get_functions(U_shape, mu_dist, K, N):
    n = len(U_shape)
    return {k: get_functions_k(k, mu_dist, U_shape, N) for k in np.ndindex(*[K]*n)}

def calculate_ergodicity(k_bands, lambd, c_k, mu):
    e = 0
    for k in k_bands:
        e += lambd[k]*(c_k[k]-mu[k])**2
    return e


# %%
class Agent:
    def __init__(self, index, N, position, U_shape, m, umax, k_bands, eps=1e-5): #mm_order=1
        """
        index : number identifier, used in adjacency matrix
        position : initial position, dim n
        U_shape : movement space
        N : number of agents
        umax : max control
        k_bands : the bands it listens to
        eps : boundary distances
        mm_order : 1 for first order (default), 2 for second order
        """
        self.index = index # identifier
        self.x_log = [position] # position log
        self.v_log = [np.zeros(len(U_shape))] # velocity log
        self.u_log = [np.zeros(len(U_shape))] # control log

        self.U_shape = U_shape
        self.n = len(U_shape)
        self.totalNum = N   # total number of agents
        self.N = N
        self.m = m
        e_init = 0
        for k in k_bands:
            e_init += lambd[k]*(fourier[k](torch.tensor(position)).numpy()-mu[k])**2
        self.e_log = [e_init] # ergodicity log

        self.umax = umax 
        # if mm_order == 2:
        #     self.vmax = self.umax
        #     self.umax = 10*self.umax
        #     self.c = 0.2 # weighting for second order dynamics
        
        # self.mm_order = mm_order
        self.k_bands = k_bands
        self.eps = eps

        
        self.c_k = {k : fourier[k](torch.tensor(position)).numpy() for k in k_bands} # shared
        self.c_k_log = [self.c_k]
        self.agent_c_k = {k : fourier[k](torch.tensor(position)).numpy() for k in k_bands} # used for calculating overall ergodicity
        # when communicating, just update the old c_k with the shared value
        # NOT averaged

    def update_shared_c_k(self, new_values):
        self.c_k = new_values
        self.c_k_log.append(self.c_k)

    def update_adj_neighbors(self, new_value):
        self.N = new_value

    def recalculate_c_k(self, t, delta_t, x_pred=None, prev_x=None, curr_x=None, old_c_k=None, old_agent_c_k=None):
        """ 
        x_pred is if want to see how c_k will change if we go to predicted value
            but don't actually want to change anything
        """
        if x_pred is None and prev_x is None and curr_x is None:
            prev_x = torch.tensor(self.x_log[-2])
            curr_x = torch.tensor(self.x_log[-1])
            fourier_fn = fourier
            new_c_k = self.c_k
            new_agent_c_k = self.agent_c_k
            old_c_k = self.c_k
            old_agent_c_k = self.agent_c_k
        elif x_pred is not None: # one time
            prev_x = casadi.MX(self.x_log[-1])
            curr_x = x_pred
            fourier_fn = casadi_fourier
            new_c_k = {}
            new_agent_c_k = {}
            old_agent_c_k = self.agent_c_k
            old_c_k = self.c_k
        else: # multiple times
            fourier_fn = casadi_fourier
            new_c_k = {}
            new_agent_c_k = {}


        for k in self.k_bands:
            average_f_k = (1/2)*(fourier_fn[k](prev_x) + fourier_fn[k](curr_x))
            if x_pred is None and prev_x is None and curr_x is None:
                average_f_k = average_f_k.numpy()
            new_c_k[k] = (old_c_k[k]*t + average_f_k*delta_t)/(t+delta_t)
            new_agent_c_k[k] = (old_agent_c_k[k]*t + average_f_k*delta_t)/(t+delta_t)

        return new_c_k, new_agent_c_k
    
    def control_casadi(self, t, delta_t):
        c_opti = casadi.Opti()
        self.c_opti = c_opti
        u = c_opti.variable(self.m)

        c_opti.subject_to( u <= self.umax )
        c_opti.subject_to( u >= -self.umax )
        # check to make sure new move is within bounds
        x_pred = self.move(u, t, delta_t, is_pred=True)
        c_opti.subject_to( x_pred >= 0 )
        c_opti.subject_to( x_pred <= np.array(self.U_shape) )
        c_opti.set_initial(u, self.u_log[-1])
        
        c_k_pred, _ = self.recalculate_c_k(t, delta_t, x_pred=x_pred)
        c_opti.minimize(calculate_ergodicity(self.k_bands, lambd, c_k_pred, mu))

        p_opts = {}
        s_opts = {'print_level': 0}
        c_opti.solver('ipopt', p_opts, s_opts)
        sol = c_opti.solve() 
        return sol.value(u)

    def control(self, t, delta_t, c_k_pred = None):
        """ returns new control """
        return NotImplemented
    def move(self, u, t, delta_t, is_pred=False, prev_x=None, prev_v=None):
        """ returns new movement given u control"""
        return NotImplemented
    def update_move(self, m):
        """ updates u_log, x_log based off of u and movement """
        return NotImplemented

    def apply_dynamics(self, t, delta_t, u=None):
        if u is None:
            u = self.control(t, delta_t)
        x = self.move(u, t, delta_t)
        self.u_log.append(u)
        self.update_move(x)

        self.recalculate_c_k(t, delta_t)
        e = calculate_ergodicity(self.k_bands, lambd, self.c_k, mu)
        self.e_log.append(e)


    def apply_dynamics_old(self, t, delta_t):
        """ Prior to apply_dynamics is communication step  -- updates c_k and Neighbors """
        """ Movement and Control Calculation """
        B = torch.zeros(self.n) 
        for k in self.k_bands: # np.ndindex(*[K]*n):
            S_k = self.totalNum*t*(self.c_k[k] - mu[k])
            grad_f_k = grad_fourier[k]
            B += lambd[k]*S_k*grad_f_k(self.x_log[-1])   

        if self.mm_order == 2:
            B = self.c*self.v_log[-1] + B

        if (torch.norm(B) < self.eps 
            or (self.x_log[-1] < self.eps).any() 
            or (torch.tensor(self.U_shape) - self.x_log[-1] < self.eps).any()):
            print("Agent: ", self.index, " had oh no at time ", t)
            
            U_center = torch.tensor(self.U_shape)*0.5
            if (U_center == self.x_log[-1]).all():
                # move in random direction away from center 
                assert(self.x_log[-1].size == self.n)
                B = torch.zeros(self.n)
                r_idx = 0
                # r_idx = random.randrange(self.n)
                B[r_idx] = 1
                
            else:
                # aim towards center if not already at center
                B = -1 * (U_center - self.x_log[-1])
        " Control "
        u = -self.umax* B / torch.norm(B)

        if self.mm_order == 1:
            x_new = self.x_log[-1] + u*delta_t
            v_new = u 
        elif self.mm_order == 2:
            x_new = self.x_log[-1] + self.v_log[-1]*delta_t + u*(delta_t**2)/2
            v_new = self.v_log[-1] + u*delta_t
            if torch.norm(v_new) > self.vmax:
                v_new = v_new/torch.norm(v_new)*self.vmax

        self.u_log.append(u)
        self.v_log.append(v_new)
        self.x_log.append(x_new)

        """ Calculate c_k"""
        for k in self.k_bands:
            average_f_k = (1/2)*(fourier[k](self.x_log[-2]) + fourier[k](self.x_log[-1]))
            self.c_k[k] = (self.c_k[k]*t + average_f_k*delta_t)/(t+delta_t)

            self.agent_c_k[k] = (self.agent_c_k[k]*t + average_f_k*delta_t)/(t+delta_t)

        """ Calculate local ergodicity -- before c_k are shared or after shared maybe should 
            wait for communication"""
        e = 0
        for k in self.k_bands:
            e += lambd[k]*(self.c_k[k]-mu[k])**2
        self.e_log.append(e)

class mm_Agent(Agent):
    def __init__(self, index, N, position, U_shape, umax, k_bands, eps=1e-5, mm_order=1, casadi_control=False):
        super().__init__(index, N, position, U_shape, n, umax, k_bands, eps=eps)
        assert(mm_order == 1 or mm_order == 2)
        self.mm_order = mm_order 
        if mm_order == 2:
            self.vmax = self.umax
            self.umax = 10*self.umax
            self.c = 0.2
        self.casadi_control = casadi_control

    def control_mm(self, t, delta_t, c_k_pred = None):
        c_k = self.c_k 
        norm = np.linalg.norm

        B = np.zeros(self.n) 
        for k in self.k_bands: # np.ndindex(*[K]*n):
            S_k = self.totalNum*t*(c_k[k] - mu[k])
            grad_f_k = grad_fourier[k]
            B += (lambd[k]*S_k*grad_f_k(torch.tensor(self.x_log[-1])).detach()).numpy()   

        if self.mm_order == 2:
            B = self.c*self.v_log[-1] + B

        if (norm(B) < self.eps 
            or (self.x_log[-1] < self.eps).any() 
            or (np.array(self.U_shape) - self.x_log[-1] < self.eps).any()):
            print("Agent: ", self.index, " had oh no at time ", t)
            
            U_center = np.array(self.U_shape)*0.5
            if (U_center == self.x_log[-1]).all():
                # move in random direction away from center 
                assert(self.x_log[-1].size == self.n)
                B = np.zeros(self.n)
                r_idx = 0
                # r_idx = random.randrange(self.n)
                B[r_idx] = 1
                
            else:
                # aim towards center if not already at center
                B = -1 * (U_center - self.x_log[-1])
        " Control "
        u = -self.umax * B / norm(B)
        return u 

    def control(self, t, delta_t, c_k_pred = None):
        if self.casadi_control:
            return self.control_casadi(t, delta_t)
        else:
            return self.control_mm(t, delta_t, c_k_pred=c_k_pred)

    def move(self, u, t, delta_t, is_pred=False, prev_x=None, prev_v=None):
        norm = np.linalg.norm
        if prev_x is None:
            prev_x = self.x_log[-1]
        if prev_v is None:
            prev_v = self.v_log[-1]

        if is_pred:
            norm = casadi.norm_2
            prev_x = casadi.MX(prev_x)
            if self.mm_order == 2:
                prev_v = casadi.MX(prev_v)

        if self.mm_order == 1:
            x_new = prev_x + u*delta_t
            v_new = u 
        elif self.mm_order == 2:
            x_new = prev_x + prev_v*delta_t + u*(delta_t**2)/2
            v_new = prev_v + u*delta_t
            # if norm(v_new) > self.vmax:
            #     v_new = v_new/norm(v_new)*self.vmax
            # assert(norm(v_new) != 0)
            # NEW
            v_new = v_new/norm(v_new)*self.vmax
        else:
            return NotImplemented

        # if is_pred:
        #     return x_new 
        # else:
        return x_new, v_new

    def update_move(self, m):
        x_new, v_new = m
        self.x_log.append(x_new)
        self.v_log.append(v_new)

# casadi optimal first/second order agent???

def runge_kutta(f, x_0, u, t, delta_t):
    # f, u are functions
    k_1 = delta_t*f(x_0, u(t))
    k_2 = delta_t*f(x_0 + 0.5*k_1, u(t + 0.5*delta_t))
    k_3 = delta_t*f(x_0 + 0.5*k_2, u(t + 0.5*delta_t))
    k_4 = delta_t*f(x_0 + k_3, u(t + delta_t))
    return x_0 + k_1/6 + k_2/3 + k_3/3 + k_4/6

class linear_Agent(Agent):
    def __init__(self, index, N, position, U_shape, umax, k_bands, A, B, m, eps=1e-5):
        super().__init__(index, N, position, U_shape, m, umax, k_bands, eps=eps)
        # x'(t) = Ax(t) + Bu(t)
        
        self.A = A # nxn
        self.u_log = [np.zeros(self.m)]
        self.B = B # nxm
        self.c_opti = None

    def control(self, t, delta_t):
        c_opti = casadi.Opti()
        self.c_opti = c_opti
        u = c_opti.variable(self.m)
        ## likely this issue, can prob easily be fixed component wise thing
        # NEWNEW first remove umax constraint
        c_opti.subject_to( u <= self.umax )
        c_opti.subject_to( u >= -self.umax )
        # check to make sure new move is within bounds
        x_pred = self.move(u, t, delta_t, is_pred=True)
        c_opti.subject_to( x_pred >= 0 )
        c_opti.subject_to( x_pred <= np.array(self.U_shape) )
        c_opti.set_initial(u, self.u_log[-1])
        
        c_k_pred, _ = self.recalculate_c_k(t, delta_t, x_pred=x_pred)
        c_opti.minimize(calculate_ergodicity(self.k_bands, lambd, c_k_pred, mu))

        p_opts = {}
        s_opts = {'print_level': 0}
        c_opti.solver('ipopt', p_opts, s_opts)
        sol = c_opti.solve() 
        return sol.value(u)

    def move(self, u, t, delta_t, is_pred=False, prev_x=None, prev_v=None):
        # x'(t) = Ax(t) + Bu(t) assumes u(t) = u(t+delta_t)
        # x(t + delta_t) = x(t) +  delta_t*(Ax(t) + Bu(t))
        prev_x = self.x_log[-1]
        A = self.A
        B = self.B
        if is_pred:
            prev_x = casadi.MX(prev_x)
            A = casadi.MX(A)
            B = casadi.MX(B)
        else:
            u = np.array(u).reshape(self.m)
         
        x_new = prev_x + delta_t*(A@prev_x + B@u)

        return x_new

    def update_move(self, m):
        x_new = m
        self.x_log.append(x_new)


# %%
""" Probability Distribution """
def normalize_mu(p):
    p_total, _ = integrate.nquad(lambda *x: p(np.array(x)), [[0, 1], [0, 1]])
    print("p_total: ", p_total)
    return lambda x: p(x)/p_total

def line(x, p1, p2, r):
    p12 = p2-p1
    mid = (p1+p2)/2
    n = np.array([-p12[1], p12[0]])
    n = n / np.linalg.norm(n)
    dist_p12 = np.dot((x-p1), n)**2
    dist_p1 = sum((x-p1)**2)
    dist_p2 = sum((x-p2)**2)
    if np.dot((x-p1), p12) > 0 and np.dot((x-p2), -p12) > 0:
        dist_seg = dist_p12
    else:
        dist_seg = min(dist_p1, dist_p2)
    return np.exp(r * dist_seg)

def unnorm_p1(x): 
    return (np.exp(-50.5 * np.sum((x[:2] - 0.2)**2)) 
    + np.exp(-50.5 * np.sum((x[:2] - 0.75)**2))                 
    + np.exp(-50.5 * np.sum((x[:2] - np.array([0.2, 0.75]))**2)))
p1 = normalize_mu(unnorm_p1)

def unnorm_p2(x): 
    return (np.exp(-50.5 * np.sum((x[:2] - 0.3)**2))                 
    + np.exp(-50.5 * np.sum((x[:2] - 0.65)**2))                 
    + np.exp(-50.5 * np.sum((x[:2] - np.array([0.3, 0.55]))**2)))
p2 = normalize_mu(unnorm_p2)

def unnorm_p3(x):
    return (np.exp(-25 * np.sum((x[:2] - 0.3)**2))             
    + np.exp(-100 * np.sum((x[:2] - 0.65)**2))             
    + np.exp(-50.5 * np.sum((x[:2] - np.array([0.3, 0.55]))**2))             
    + np.exp(-200 * np.sum((x[:2] - 0.2)**2))             
    + np.exp(-50 * np.sum((x[:2] - 0.75)**2))             
    + np.exp(-150 * np.sum((x[:2] - np.array([0.2, 0.75]))**2))             
    + np.exp(-150 * np.sum((x[:2] - np.array([0.5, 0.5]))**2))             
    + np.exp(-300 * np.sum((x[:2] - np.array([0.8, 0.2]))**2))             
    + np.exp(-200 * np.sum((x[:2] - np.array([0.9, 0.3]))**2)))
p3 = normalize_mu(unnorm_p3)

def unnorm_p4(x):
    return (line(x, np.array([0.2, 0.7]), np.array([0.7, 0.2]), -100)             
    + line(x, np.array([0.1, 0.1]), np.array([0.9, 0.9]), -50)             
    + np.exp(-200 * np.sum((x[:2] - np.array([0.9, 0.4]))**2))             
    + np.exp(-100 * np.sum((x[:2] - np.array([0.4, 0.8]))**2))             
    + np.exp(-50 * np.sum((x[:2] - np.array([0.2, 0.5]))**2))) 
p4 = normalize_mu(unnorm_p4)

# Grid and Display
X,Y = np.meshgrid(*[np.linspace(0,1)]*2)
_s = np.stack([X.ravel(), Y.ravel()]).T


plt.contourf(X, Y, np.array(list(map(p1, _s))).reshape(X.shape))
plt.contourf(X, Y, np.array(list(map(p2, _s))).reshape(X.shape)) 
plt.contourf(X, Y, np.array(list(map(p3, _s))).reshape(X.shape))      
plt.contourf(X, Y, np.array(list(map(p4, _s))).reshape(X.shape))    



# %%
""" Parameters """
# x = [np.array([0.54,0.3]), np.array([0.2, 0.7]), np.array([0.8, 0.1]),
#      np.array([0.7,0.3]), np.array([0.3, 0.7]), np.array([0.9, 0.3]),
#      np.array([0.1,0.2]), np.array([0.2, 0.3]), np.array([0.9, 0.8])]
x = [np.array([0.54,0.3]), np.array([0.2, 0.7]), np.array([0.8, 0.1]),
     np.array([0.7,0.3]), np.array([0.3, 0.7]), np.array([0.9, 0.3]),
     np.array([0.1,0.2]), np.array([0.2, 0.3]), np.array([0.9, 0.8]), np.array([0.4, 0.5])]

U_shape=(1, 1)
n = len(U_shape)
mu_dist=p3
plt.contourf(X, Y, np.array(list(map(mu_dist, _s))).reshape(X.shape)) 

N=len(x)
delta_t=0.1
K=5
u_max=0.5
num_iter = 2000
num_plan = 100
sparseweight = 0.1
O = np.ones((3, 3)) # A = 1 full communication
I = np.diag(np.ones(3)) # A = Id isolation
P = np.array([[1, 1, 0], # A = P
                  [1, 1, 0],
                  [0, 0, 1]])

C = np.array([[1, 1, 0], # A = C
                  [1, 1, 1],
                  [0, 1, 0]]) # some weird pairing
# Also A = Lr : communicates with things within radius r
A = P

casadi_fourier = {}
fourier = {}
grad_fourier = {}
lambd = {}
mu = {}
for k in np.ndindex(*[K]*n):
    print(k)
    new_k = np.array(k)*np.pi/np.array(U_shape)
    casadi_fourier[k] = get_casadi_fourier_fn_k(new_k, U_shape)
    fourier[k] = get_fourier_fn_k(new_k, U_shape)
    grad_fourier[k] = get_grad_fourier_fn_k(fourier[k])
    lambd[k] = get_lambda_k(new_k, n)
    mu[k] = get_mu_k(mu_dist, fourier[k], U_shape)


# %%
# filename="mm_ergodic_order2_casadi" 
# agents = [mm_Agent(i, N, x[i], U_shape, u_max, list(np.ndindex(*[K]*n)), mm_order=2) for i in range(N)]
filename="multiagents{}_p3_toy_comparison_nocomm_timesteps_{}_delta_{}_sparseweight_{}".format(N, num_plan, delta_t, sparseweight) 
# A = [np.array([[0, -0.5], [0.5, 0]]), np.array([[-0.5, 0], [0, -0.5]]), np.array([[-0.5, 0.5], [0.5, -0.5]])]
# B = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]
# m = [len(b) for b in B]
# # for linear, umax should be bigger than the max value of the vector field at the very least
# linear_agents = [linear_Agent(i, N, x[i], U_shape, 2, list(np.ndindex(*[K]*n)), A[i], B[i], m[i]) for i in range(3)]
# mm1_agents = [mm_Agent(3+i, N, x[3+i], U_shape, u_max, list(np.ndindex(*[K]*n)), mm_order=1) for i in range(3)]
# mm2_agents = [mm_Agent(6+i, N, x[6+i], U_shape, u_max, list(np.ndindex(*[K]*n)), mm_order=2) for i in range(3)]
# agents = linear_agents + mm1_agents + mm2_agents
print("filename: {}".format(filename))
mm1_agents = [mm_Agent(i, N, x[i], U_shape, u_max, list(np.ndindex(*[K]*n)), mm_order=1, casadi_control=False) for i in range(5)]
mm2_agents = [mm_Agent(5+i, N, x[5+i], U_shape, u_max, list(np.ndindex(*[K]*n)), mm_order=2, casadi_control=False) for i in range(5)]
agents = mm1_agents + mm2_agents


mm1_agents_central = [mm_Agent(i, N, x[i], U_shape, u_max, list(np.ndindex(*[K]*n)), mm_order=1, casadi_control=False) for i in range(5)]
mm2_agents_central = [mm_Agent(5+i, N, x[5+i], U_shape, u_max, list(np.ndindex(*[K]*n)), mm_order=2, casadi_control=False) for i in range(5)]
agents_central = mm1_agents_central + mm2_agents_central

# %%



# %%
def oneminussqr_metric(x):     
    return 1 - (x-1)**2
def oneminusoneover_metric(x, a): 
    return 1 - 1/(a*(x+1/a))  


elt_metric = lambda x: oneminusoneover_metric(x, 100)
r = 2
""" WEIGHED VERSION """
overall_e_log = []
A_log = []
planned_e_log = []

""" Casadi Optimization -- Optimal Adjacency Matrix """
opti = casadi.Opti()

""" Unrolling of planning horizon """
u_opt_plan = []
curr_x_plan = [[agents[j].x_log[-1]] for j in range(N)]
curr_v_plan = [[np.zeros(agents[j].m)] for j in range(N)]
curr_c_k_plan = [[agents[j].c_k] for j in range(N)]
curr_agent_c_k_plan = [[agents[j].agent_c_k] for j in range(N)]
u_opt_plan = []
for j in range(N):
    u_j = opti.variable(num_plan, agents[j].m)
    init = np.array([[agents[j].umax*int(agents[j].m-1 == x) for x in range(agents[j].m)]])
    opti.set_initial(u_j, np.repeat(init, [num_plan], axis=0))
    u_opt_plan.append(u_j)

for j in range(N):
    for plan_i in range(num_plan):
        opti.subject_to(casadi.sum1(u_opt_plan[j][plan_i,:]**2) <= agents[j].umax**2)

for plan_i in range(0, num_plan):
    # i = big_i*num_plan + plan_i
    i = plan_i
    t = i*delta_t # [time, time+time_step]

    for j in range(N):        
        x, v = agents[j].move(u_opt_plan[j][plan_i,:].T, t, delta_t, is_pred=True, 
                                    prev_x=curr_x_plan[j][-1], prev_v=curr_v_plan[j][-1])
        curr_x_plan[j].append(x)
        curr_v_plan[j].append(v)
        opti.subject_to(curr_x_plan[j][-1][:] >= 0)
        opti.subject_to(curr_x_plan[j][-1][:] <= np.array(agents[j].U_shape))
        curr_c_k_j, curr_agent_c_k_j = agents[j].recalculate_c_k(t, delta_t, prev_x=curr_x_plan[j][-2], curr_x=curr_x_plan[j][-1], 
                                            old_c_k=curr_c_k_plan[j][-1], old_agent_c_k=curr_agent_c_k_plan[j][-1])
        curr_c_k_plan[j].append(curr_agent_c_k_j)
        curr_agent_c_k_plan[j].append(curr_agent_c_k_j)
    
A_opt = opti.variable(1, N)
opti.subject_to(casadi.vec(A_opt) >= 0)                       # elements nonnegative
opti.subject_to(casadi.vec(A_opt) <= 1)                       # elements bounded by 1
opti.subject_to(A_opt@np.ones(N) == 1)                        # rows sum to 1


opti.set_initial(A_opt, np.array([int(j==0) for j in range(N)]))
ck_new = {}
for k in np.ndindex(*[K]*n):
    ck = np.array([agents[j].c_k[k] for j in range(N)])
    # ck_sum = casadi.sum1(ck)*np.ones(N)
    # can try abs sum also
    ck_new[k] = A_opt@ck
     
# Sparse Metric -- Average element penalization
sparsemetric = casadi.sum1(casadi.sum2(elt_metric(A_opt)))/A_opt.numel()  
# opti.subject_to(sparsemetric < 0.25)

e_plan = 0
for k in np.ndindex(*[K]*n):
    e_plan += lambd[k]*(ck_new[k]-mu[k])**2

# readjust sparsemetric range to be [0, e_plan]
opti.minimize(e_plan*(1 + sparsemetric))


p_opts = {}
s_opts = {'print_level': 0}
opti.solver('ipopt', p_opts, s_opts)
sol = opti.solve()

A = sol.value(A_opt).round(r)
u_plan = [sol.value(u_opt_plan[j]) for j in range(N)]
x_plan = [[sol.value(curr_x_plan[j][plan_i]) for plan_i in range(num_plan)] for j in range(N)]
print("e_plan: ", sol.value(e_plan))


# %%
A_log.append(A.reshape(1, N))
planned_e_log = []
" Applying individual controls"
for plan_i in range(0, num_plan):
    for j in range(N):    
        agents[j].apply_dynamics(t, delta_t, u=u_plan[j][plan_i]) # here

        

    """ Calculate overall ergodicity """
    e = 0
    plan_e = 0
    for k in np.ndindex(*[K]*n):
        ave_c_k = (1/N)*sum([agents[j].agent_c_k[k] for j in range(N)])
        e += lambd[k]*(ave_c_k-mu[k])**2

        plan_e += lambd[k]*(A_log[0]@np.array([agents[j].agent_c_k[k] for j in range(N)])-mu[k])**2
    overall_e_log.append(plan_e)   ###############
    planned_e_log.append(plan_e)


""" CENTRAL VERSION """
overall_e_log_central = []
A_log_central = []

# """ Casadi Optimization -- Optimal Adjacency Matrix """
# opti_central = casadi.Opti()

# """ Unrolling of planning horizon """
# u_opt_plan_central = []
# curr_x_plan_central = [[agents_central[j].x_log[-1]] for j in range(N)]
# curr_v_plan_central = [[np.zeros(agents_central[j].m)] for j in range(N)]
# curr_c_k_plan_central = [[agents_central[j].c_k] for j in range(N)]
# curr_agent_c_k_plan_central = [[agents_central[j].agent_c_k] for j in range(N)]
# u_opt_plan_central = []
# for j in range(N):
#     u_j_central = opti_central.variable(num_plan, agents_central[j].m)
#     init_central = np.array([[agents_central[j].umax*int(agents_central[j].m-1 == x) for x in range(agents_central[j].m)]])
#     opti_central.set_initial(u_j_central, np.repeat(init_central, [num_plan], axis=0))
#     u_opt_plan_central.append(u_j_central)

# for j in range(N):
#     for plan_i in range(num_plan):
#         opti_central.subject_to(casadi.sum1(u_opt_plan_central[j][plan_i,:]**2) < agents_central[j].umax**2)

# for plan_i in range(0, num_plan):
#     # i = big_i*num_plan + plan_i
#     i = plan_i
#     t = i*delta_t # [time, time+time_step]

#     for j in range(N):        
#         x, v = agents_central[j].move(u_opt_plan_central[j][plan_i,:].T, t, delta_t, is_pred=True, 
#                                     prev_x=curr_x_plan_central[j][-1], prev_v=curr_v_plan_central[j][-1])
#         curr_x_plan_central[j].append(x)
#         curr_v_plan_central[j].append(v)
#         opti_central.subject_to(curr_x_plan_central[j][-1][:] > 0)
#         opti_central.subject_to(curr_x_plan_central[j][-1][:] < np.array(agents_central[j].U_shape))
#         curr_c_k_j_central, curr_agent_c_k_j_central = agents_central[j].recalculate_c_k(t, delta_t, prev_x=curr_x_plan_central[j][-2], curr_x=curr_x_plan_central[j][-1], 
#                                             old_c_k=curr_c_k_plan_central[j][-1], old_agent_c_k=curr_agent_c_k_plan_central[j][-1])
#         curr_c_k_plan_central[j].append(curr_agent_c_k_j_central)
#         curr_agent_c_k_plan_central[j].append(curr_agent_c_k_j_central)
    
# A_opt_central = np.ones(N).reshape(1, N) / N

# ck_new_central = {}
# for k in np.ndindex(*[K]*n):
#     ck_central = np.array([agents_central[j].c_k[k] for j in range(N)])
#     # ck_sum = casadi.sum1(ck)*np.ones(N)
#     # can try abs sum also
#     ck_new_central[k] = A_opt_central@ck_central
     
# # Sparse Metric -- Average element penalization
# sparsemetric_central = 1  
# # opti.subject_to(sparsemetric < 0.25)

# e_plan_central = 0
# for k in np.ndindex(*[K]*n):
#     e_plan_central += lambd[k]*(ck_new_central[k]-mu[k])**2

# # readjust sparsemetric range to be [0, e_plan]
# opti_central.minimize(e_plan_central*(1 + sparsemetric_central))


# p_opts = {}
# s_opts = {'print_level': 0}
# opti_central.solver('ipopt', p_opts, s_opts)
# sol_central = opti_central.solve()

# A_central = A_opt_central
# u_plan_central = [sol_central.value(u_opt_plan_central[j]) for j in range(N)]
# x_plan_central = [[sol_central.value(curr_x_plan_central[j][plan_i]) for plan_i in range(num_plan)] for j in range(N)]
# print("e_plan_central: ", sol_central.value(e_plan_central))


# %%
A_central = np.ones(N).reshape(1, N) / N
A_log_central.append(A_central.reshape(1, N))

" Applying individual controls"
for plan_i in range(0, num_plan):
    for j in range(N):    
        agents_central[j].apply_dynamics(t, delta_t)#, u=u_plan_central[j][plan_i]) # here

        

    """ Calculate overall ergodicity """
    e_central = 0

    for k in np.ndindex(*[K]*n):
        ave_c_k_central = (1/N)*sum([agents_central[j].agent_c_k[k] for j in range(N)])
        e_central += lambd[k]*(A_log_central[0]@np.array([agents_central[j].agent_c_k[k] for j in range(N)])-mu[k])**2
    overall_e_log_central.append(e_central)  





# %%
print("control plan for agent 0")
print(u_plan[0][:5])
print("location plan for agent 0")
print(x_plan[0][:5])
print("controls to get location")
print([np.zeros(2) if i == 0 else (x_plan[0][i] - x_plan[0][i-1])/delta_t for i in range(5)])


# print("control plan_central for agent 0")
# print(u_plan_central[0][:5])
# print("location plan_central for agent 0")
# print(x_plan_central[0][:5])
# print("controls_central to get location")
# print([np.zeros(2) if i == 0 else (x_plan_central[0][i] - x_plan_central[0][i-1])/delta_t for i in range(5)])

# %%
""" Animate """ 
# colors = ['r', 'm', 'c', 'y', 'g', 'b', 'k', 'w']
colors = ['maroon', 'cyan', 'black', 'slateblue', 'orange', 'indigo', 'magenta', 'pink', 'yellow', 'green']
if N > len(colors):
    raise NotYetImplemented("Number of agents: ", N, "- Animation of this dimension is not supported.")

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.4*3, 4.8))
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(6.4*2, 4.8*2))

fig.suptitle('Ergodic Coverage')


ax1.set_title('Weighted Trajectory')
ax1.set_aspect('equal')
ax1.set_xlim(0, U_shape[0])
ax1.set_ylim(0, U_shape[1])

ax4.set_title('Central Trajectory')
ax4.set_aspect('equal')
ax4.set_xlim(0, U_shape[0])
ax4.set_ylim(0, U_shape[1])

X,Y = np.meshgrid(np.linspace(0, U_shape[0]), np.linspace(0, U_shape[1]))
_s = np.stack([X.ravel(), Y.ravel()]).T
ax1.contourf(X, Y, np.array(list(map(mu_dist, _s))).reshape(X.shape))
ax4.contourf(X, Y, np.array(list(map(mu_dist, _s))).reshape(X.shape))

ax2.set_title('Weighted Local Ergodicities')
max_local_e = float(max([max(agent.e_log) for agent in agents]))
max_e = max(float(max(overall_e_log)), float(max(planned_e_log)), max_local_e)
ax2.set_ylim(0, max_e)
ax2.set(xlabel='Time')

ax5.set_title('Central Local Ergodicities')
max_local_e_central = float(max([max(agent.e_log) for agent in agents_central]))
max_e_central = max(float(max(overall_e_log_central)), max_local_e_central)
ax5.set_ylim(0, max_e_central)
ax5.set(xlabel='Time')

ax3.set_title('Weight Matrix')

ax6.set_title('Comparison of Ergodicities')
ax6.set_ylim(0, max(overall_e_log + overall_e_log_central))
ax6.set(xlabel='Time')

fig.tight_layout()

pos_data = [([], []) for i in range(N)]
pos_data_central = [([], []) for i in range(N)]
time_data = []
ergodicity_data = []
ergodicity_data_central = []
local_ergodicity_data = [[] for i in range(N)]
local_ergodicity_data_central = [[] for i in range(N)]
pos_lns = []
pos_lns_central = []
local_erg_lns = []
local_erg_lns_central = []


for i in range(N):
    pos_ln, = ax1.plot(pos_data[i][0], pos_data[i][1], c=colors[i], label=i)
    pos_lns.append(pos_ln)
    pos_ln_central, = ax4.plot(pos_data_central[i][0], pos_data_central[i][1], c=colors[i], label=i)
    pos_lns_central.append(pos_ln_central)

    local_erg, = ax2.plot(time_data, local_ergodicity_data[i], c=colors[i], label=i)
    local_erg_lns.append(local_erg)
    local_erg_central, = ax5.plot(time_data, local_ergodicity_data_central[i], c=colors[i], label=i)
    local_erg_lns_central.append(local_erg_central)

ergodicity_ln, = ax2.plot(time_data, ergodicity_data, c='b')
ergodicity_ln_central, = ax5.plot(time_data, ergodicity_data_central, c='r')

consensus_ln = ax3.matshow(A_log[0], vmin=0, vmax=1, cmap="Greys")
combined_erg_ln_weight, = ax6.plot(time_data, ergodicity_data, c='b')
combined_erg_ln_central, = ax6.plot(time_data, ergodicity_data_central, c='r')



def animate2d_init():
    time_max = len(overall_e_log)
    ax2.set_xlim(0, time_max)
    ax5.set_xlim(0, time_max)
    ax6.set_xlim(0, time_max)
    return (*pos_lns, *pos_lns_central, *local_erg_lns, *local_erg_lns_central, ergodicity_ln, ergodicity_ln_central, 
            consensus_ln, combined_erg_ln_weight, combined_erg_ln_central)

def animate2d_from_logs_update(frame):
    time_data.append(frame)
    idx = 0
    for i in range(N):
        pos_data[i][0].append(agents[i].x_log[frame][0])
        pos_data[i][1].append(agents[i].x_log[frame][1])
        pos_lns[i].set_data(pos_data[i][0], pos_data[i][1])

        pos_data_central[i][0].append(agents_central[i].x_log[frame][0])
        pos_data_central[i][1].append(agents_central[i].x_log[frame][1])
        pos_lns_central[i].set_data(pos_data_central[i][0], pos_data_central[i][1])
        
        local_ergodicity_data[i].append(agents[i].e_log[frame])
        local_erg_lns[i].set_data(time_data, local_ergodicity_data[i])

        local_ergodicity_data_central[i].append(agents_central[i].e_log[frame])
        local_erg_lns_central[i].set_data(time_data, local_ergodicity_data_central[i])
        
    
    ergodicity_data.append(overall_e_log[frame])
    ergodicity_ln.set_data(time_data, ergodicity_data)

    ergodicity_data_central.append(overall_e_log_central[frame])
    ergodicity_ln_central.set_data(time_data, ergodicity_data_central)

    consensus_ln.set_data(A_log[frame//num_plan])
    
    combined_erg_ln_weight.set_data(time_data, ergodicity_data)
    combined_erg_ln_central.set_data(time_data, ergodicity_data_central)

    return (*pos_lns, *pos_lns_central, *local_erg_lns, *local_erg_lns_central, ergodicity_ln, ergodicity_ln_central, 
            consensus_ln, combined_erg_ln_weight, combined_erg_ln_central)

update = animate2d_from_logs_update
frames = len(overall_e_log)


FFwriter = animation.writers['ffmpeg']
writer = FFwriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
anime = animation.FuncAnimation(fig, animate2d_from_logs_update, init_func=animate2d_init, 
                            frames=frames, interval=20, blit=True)  
plt.show()
if filename is not None:
    anime.save(filename+".mp4", writer=writer) 
