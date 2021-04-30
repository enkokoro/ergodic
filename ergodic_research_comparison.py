# Comparison of weighted and equally weighted communication
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

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
    def __init__(self, index, N, position, U_shape, m, umax, k_bands, eps=1e-5):
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
        is_not_prediction = x_pred is None and prev_x is None and curr_x is None
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
            if is_not_prediction:
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
        c_opti.subject_to( u**2 <= self.umax**2 )
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

N=5#len(x)
delta_t=0.05
K=3
u_max=0.5
# num_iter = 2000
# num_plan = 50
num_iter = 100
num_plan = 10
sparseweight = 0.1


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

def optimize_weights(agents, agentidx, server, N, cur_iter, num_steps, elt_metric=(lambda x: oneminusoneover_metric(x, 100))):
    # optimizes all controls such that final ergodicity calculated for all agents is low
    """ Casadi Optimization -- Optimal Adjacency Matrix """
    opti = casadi.Opti()

    """ Unrolling of planning horizon """
    curr_x_plan = [[agents[j].x_log[-1]] for j in range(N)]
    curr_v_plan = [[np.zeros(agents[j].m)] for j in range(N)]
    curr_c_k_plan = [[server[j]] for j in range(N)]
    curr_agent_c_k_plan = [[server[j]] for j in range(N)]
    u_opt_plan = []
    for j in range(N):
        u_j = opti.variable(num_steps, agents[j].m)
        init = np.array([[agents[j].umax*int(agents[j].m-1 == x) for x in range(agents[j].m)]])
        opti.set_initial(u_j, np.repeat(init, [num_steps], axis=0))
        u_opt_plan.append(u_j)

    for j in range(N):
        for plan_i in range(num_steps):
            opti.subject_to(casadi.sum1(u_opt_plan[j][plan_i,:]**2) <= agents[j].umax**2)

    for plan_i in range(0, num_steps):
        i = cur_iter + plan_i
    
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
            curr_c_k_plan[j].append(curr_c_k_j)
            curr_agent_c_k_plan[j].append(curr_agent_c_k_j)
        
    A_opt = opti.variable(1, N)
    # opti.subject_to(A_opt[agentidx] >= 1/N)
    opti.subject_to(casadi.vec(A_opt) >= 0)                       # elements nonnegative
    opti.subject_to(casadi.vec(A_opt) <= 1)                       # elements bounded by 1
    opti.subject_to(A_opt@np.ones(N) == 1)                        # rows sum to 1


    opti.set_initial(A_opt, np.array([int(j==0) for j in range(N)]))
    ck_new = {}
    for k in np.ndindex(*[K]*n):
        ck = np.array([curr_c_k_plan[j][k] for j in range(N)])
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

    r = 2
    A = sol.value(A_opt).round(r)
    u_plan = [sol.value(u_opt_plan[j]) for j in range(N)]
    x_plan = [[sol.value(curr_x_plan[j][plan_i]) for plan_i in range(num_steps)] for j in range(N)]
    print("e_plan: ", sol.value(e_plan))
    if np.count_nonzero(A) == 1 and A[agentidx] != 0:
        # only communicating with itself
        A = np.zeros(N)
    return A, x_plan, u_plan

def optimize_weights2(agents, agentidx, server, N, cur_iter, num_steps, elt_metric=(lambda x: oneminusoneover_metric(x, 100))):
    # only optimizes agent's controls so that final c_k ergodicity is low
    """ Casadi Optimization -- Optimal Adjacency Matrix """
    opti = casadi.Opti()

    A_opt = opti.variable(1, N)
    opti.subject_to(casadi.vec(A_opt) >= 0)                       # elements nonnegative
    opti.subject_to(casadi.vec(A_opt) <= 1)                       # elements bounded by 1
    opti.subject_to(A_opt@np.ones(N) == 1)                        # rows sum to 1


    opti.set_initial(A_opt, np.array([int(j==agentidx) for j in range(N)]))
    ck_new = {}
    for k in np.ndindex(*[K]*n):
        ck = np.array([server[j][k] for j in range(N)])
        ck_new[k] = A_opt@ck

        
    """ Unrolling of planning horizon """
    u_opt_plan = opti.variable(num_steps, agents[agentidx].m)
    init = np.array([[agents[agentidx].umax*int(agents[agentidx].m-1 == x) for x in range(agents[agentidx].m)]])
    opti.set_initial(u_opt_plan, np.repeat(init, [num_steps], axis=0))

    curr_x_plan = [agents[agentidx].x_log[-1]]
    curr_v_plan = [np.zeros(agents[agentidx].m)]
    curr_c_k_plan = [ck_new]
    curr_agent_c_k_plan = [agents[agentidx].agent_c_k]
    

    for plan_i in range(0, num_steps):
        i = cur_iter + plan_i
    
        t = i*delta_t # [time, time+time_step]
        
        u = u_opt_plan[plan_i,:].T    
        opti.subject_to(casadi.sum1(u_opt_plan[plan_i,:]**2) <= agents[agentidx].umax**2)  
        x, v = agents[agentidx].move(u, t, delta_t, is_pred=True, 
                                    prev_x=curr_x_plan[-1], prev_v=curr_v_plan[-1])
        curr_x_plan.append(x)
        curr_v_plan.append(v)
        opti.subject_to(curr_x_plan[-1][:] >= 0)
        opti.subject_to(curr_x_plan[-1][:] <= np.array(agents[agentidx].U_shape))
        curr_c_k, curr_agent_c_k = agents[agentidx].recalculate_c_k(t, delta_t, prev_x=curr_x_plan[-2], curr_x=curr_x_plan[-1], 
                                            old_c_k=curr_c_k_plan[-1], old_agent_c_k=curr_agent_c_k_plan[-1])
        curr_c_k_plan.append(curr_c_k)
        curr_agent_c_k_plan.append(curr_agent_c_k)
        
    
        
    # Sparse Metric -- Average element penalization
    sparsemetric = casadi.sum1(casadi.sum2(elt_metric(A_opt)))/A_opt.numel()  
    # opti.subject_to(sparsemetric < 0.25)

    e_plan = 0
    for k in np.ndindex(*[K]*n):
        ave_c_k = np.array([server[j][k] for j in range(N)])
        ave_c_k[agentidx] = curr_c_k_plan[-1][k]
        ave_c_k = sum(ave_c_k)/N
        e_plan += lambd[k]*(ave_c_k-mu[k])**2

    # readjust sparsemetric range to be [0, e_plan]
    opti.minimize(e_plan*(1 + sparsemetric))


    p_opts = {}
    s_opts = {'print_level': 0}
    opti.solver('ipopt', p_opts, s_opts)
    sol = opti.solve()

    r = 2
    A = sol.value(A_opt).round(r)
    u_plan = sol.value(u_opt_plan)
    x_plan = [sol.value(curr_x_plan[plan_i]) for plan_i in range(num_steps)]
    print("e_plan: ", sol.value(e_plan))
    
    if np.count_nonzero(A) == 1 and A[agentidx] != 0:
        # only communicating with itself
        A = np.zeros(N)
    return A, x_plan, u_plan



def local_broadcast(agents, agentidx, server, N, cur_iter, num_steps, radius=0.25):
    A = np.zeros(N)
    total = 0
    for j in range(N):
        if np.linalg.norm(agents[j].x_log[-1] - agents[agentidx].x_log[-1]) < radius:
            A[j] = 1
            total+=1
    if total == 1:
        A = np.zeros(N)
    if total != 0:
        A = A/total
    return A, [], []

def full_communication(agents, agentidx, server, N, cur_iter, num_steps):
    A = np.ones(N)/N 
    return A, [], []

# %%
def oneminussqr_metric(x):     
    return 1 - (x-1)**2
def oneminusoneover_metric(x, a): 
    return 1 - 1/(a*(x+1/a))  


elt_metric = lambda x: oneminusoneover_metric(x, 100)

def run_trajectories(agents, need_to_communicate, row_weights):
    overall_e_log = []
    A_log = []
    server = [agent.c_k.copy() for agent in agents]

    communication_cost = []
    for i in range(num_iter):
        t = i*delta_t
        A = []
        communicate = [False for j in range(N)]

        """ Communication """
        for j in range(N):
            cost = 0
            if need_to_communicate(i, agents, j, server):
                communicate[j] = True
                A_row, x_plan, u_plan = row_weights(agents, j, server, N, i, num_plan)

                for idx in range(0, N):
                    if A_row[idx] != 0 or idx == j:
                        # two because receive and send
                        cost+=2
            else:
                A_row = np.zeros(N)

            communication_cost.append(cost)
            A.append(A_row)

        for j in range(N):
            if communicate[j]:
                for k in np.ndindex(*[K]*n):
                    c_k = np.array([agent.c_k[k] for agent in agents])
                    agents[j].c_k[k] = A[j]@c_k
                server[j] = agents[j].c_k.copy()
                    # for _j in range(N):
                    #     if A[j][_j] > 0.01:
                    #         server[_j] = agents[_j].c_k.copy()
                
        A = np.stack(A)
        A_log.append(A)

        """ Movement """
        for j in range(N):   
            agents[j].apply_dynamics(t, delta_t)
        
        """ Calculate Overall Ergodicity """
        e = 0
        for k in np.ndindex(*[K]*n):
            ave_c_k = (1/N)*sum([agents[j].agent_c_k[k] for j in range(N)])
            e += lambd[k]*(ave_c_k-mu[k])**2

        overall_e_log.append(e)
    return overall_e_log, A_log, communication_cost

###################################
def c_k_error_metric(c_ks_1, c_ks_2):
    total_error = 0
    total_num = 0
    for k in np.ndindex(*[K]*n):
        total_num += 1
        total_error += abs(c_ks_1[k] - c_ks_2[k])
    print(total_error/total_num)
    return total_error/total_num
###################################
def communicate_c_k_error(agents, j, server, error):
    return c_k_error_metric(agents[j].c_k, server[j]) >= error

###################################
error = 0.25
weight_fun = 'optiweight2'
###################################
c_k_error_communication = lambda _timestep, _agents, _agentidx, _server: communicate_c_k_error(_agents, _agentidx, _server, error)
periodic_communication = lambda _timestep, _agents, _agentidx, _server: _timestep % num_plan == 0
no_communication = lambda _timestep, _agents, _agentidx, _server: False

labels = ['Adaptive', 'Periodic', 'No']

communication_conditions = [c_k_error_communication, periodic_communication, no_communication]
weight_conditions = {'optiweight': optimize_weights,
                     'optiweight2': optimize_weights2,
                     'local': local_broadcast,
                     'full': full_communication}

assert(len(labels) == len(communication_conditions))
info = []
for cc in communication_conditions:
    mm1_agents = [mm_Agent(i, N, x[i], U_shape, u_max, list(np.ndindex(*[K]*n)), mm_order=1, casadi_control=False) for i in range(5)]
    # mm2_agents = [mm_Agent(5+i, N, x[5+i], U_shape, u_max, list(np.ndindex(*[K]*n)), mm_order=2, casadi_control=False) for i in range(5)]
    agents = mm1_agents #+ mm2_agents

    overall_e_log, A_log, communication_cost = run_trajectories(agents, cc, weight_conditions[weight_fun])
    info.append({'agents': agents, 
                 'overall_e_log': overall_e_log, 
                 'A_log': A_log, 
                 'communication_cost': communication_cost})

filename="multiagents{}_p3_comparison_{}_labels_{}_error_{}_timesteps_{}_num_plan_{}_delta_{}".format(N, weight_fun, "".join(labels), error, num_iter, num_plan, delta_t) 

print("filename: {}".format(filename))

# %%
""" Animate """ 
# colors = ['r', 'm', 'c', 'y', 'g', 'b', 'k', 'w']
colors = ['maroon', 'darkorange', 'yellow', 'lawngreen', 'cyan', 'teal', 'dodgerblue', 'slateblue', 'indigo', 'magenta', 'hotpink']
e_colors = ['red', 'green', 'blue']
if N > len(colors):
    raise NotYetImplemented("Number of agents: ", N, "- Animation of this dimension is not supported.")

fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3, 5, figsize=(6.4*3, 4.8*5))

traj = [ax1,ax6,ax11]
local_erg = [ax2,ax7,ax12]
weights = [ax3,ax8,ax13]
comm = [ax4,ax9,ax14]
num_compar = 3
assert(num_compar == len(traj))
assert(num_compar == len(local_erg))
assert(num_compar == len(weights))
assert(num_compar == len(comm))
fig.suptitle('Adaptive Communication for Decentralized Ergodic Coverage')

X,Y = np.meshgrid(np.linspace(0, U_shape[0]), np.linspace(0, U_shape[1]))
_s = np.stack([X.ravel(), Y.ravel()]).T

max_local_e = float(max([max([max(agent.e_log) for agent in inf['agents']]) for inf in info]))
max_e = float(max([max(inf['overall_e_log']) for inf in info]))
max_num_comm = max([max(inf['communication_cost']) for inf in info]) + 2
max_total_num_comm = max([sum(inf['communication_cost']) for inf in info]) + 2
max_time = num_iter


traj_data = []
local_erg_data = []
overall_erg_data = []
weight_data = []
comm_data = []
total_comm_data = []
time_data = []
for i in range(num_compar):
    traj_datai = [([], []) for j in range(N)]
    traj_data.append(traj_datai)
    local_erg_datai = [[] for j in range(N)]
    local_erg_data.append(local_erg_datai)
    overall_erg_datai = []
    overall_erg_data.append(overall_erg_datai)
    weight_datai = np.zeros((N, N))
    weight_data.append(weight_datai)
    comm_datai = []
    comm_data.append(comm_datai)
    total_comm_datai = []
    total_comm_data.append(total_comm_datai)

    traj[i].set_title(labels[i]+" Comm Trajectory")
    traj[i].set_aspect('equal')
    traj[i].set_xlim(0, U_shape[0])
    traj[i].set_ylim(0, U_shape[1])
    traj[i].contourf(X, Y, np.array(list(map(mu_dist, _s))).reshape(X.shape))

    local_erg[i].set_title('Local Ergodicities')
    local_erg[i].set_xlim(0, max_time)
    local_erg[i].set_ylim(0, max_local_e)
    local_erg[i].set(xlabel='Time')

    weights[i].set_title('Consensus Matrix')

    comm[i].set_title('Number of communications')
    comm[i].set_xlim(0, max_time)
    comm[i].set_ylim(0, max_num_comm)
    comm[i].set(xlabel='Time')

ax5.set_title('Number of communications') 
ax5.set_xlim(0, max_time)
ax5.set_ylim(0, max_num_comm)
ax5.set(xlabel='Time')

ax10.set_title('Total number of communications')
ax10.set_xlim(0, max_time)
ax10.set_ylim(0, max_total_num_comm)
ax10.set(xlabel='Time')

ax15.set_title('Comparison of ergodicities')
ax15.set_xlim(0, max_time)
ax15.set_ylim(0, max_e)
ax15.set(xlabel='Time')

fig.tight_layout()

traj_lns = [[] for i in range(num_compar)]
local_erg_lns = [[] for i in range(num_compar)]
overall_erg_lns = []
weight_lns = []
comm_lns = []

compare_comm_lns = []
compare_total_comm_lns = []
compare_erg_lns = []

for i in range(num_compar):
    for j in range(N):
        traj_lnj, = traj[i].plot(traj_data[i][j][0], traj_data[i][j][1], c=colors[j], label="traj({},{})".format(i, j))
        traj_lns[i].append(traj_lnj)

        local_ergj, = local_erg[i].plot(time_data, local_erg_data[i][j], c=colors[j], label="local_erg({},{})".format(i, j))
        local_erg_lns[i].append(local_ergj)

    overall_erg_lni, = local_erg[i].plot(time_data, overall_erg_data[i], c=e_colors[i], label="overall_erg({})".format(i))
    overall_erg_lns.append(overall_erg_lni)
    weight_lni = weights[i].matshow(weight_data[i], vmin=0, vmax=1, cmap="Greys")
    weight_lns.append(weight_lni)
    comm_lni, = comm[i].plot(time_data, comm_data[i], label="comm({})".format(i))
    comm_lns.append(comm_lni)

    compare_comm_ln, = ax5.plot(time_data, comm_data[i], c=e_colors[i], label="compare_comm({})".format(i))
    compare_comm_lns.append(compare_comm_ln)
    compare_total_comm_ln, = ax10.plot(time_data, total_comm_data[i], c=e_colors[i], label="compare_total_comm({})".format(i))
    compare_total_comm_lns.append(compare_total_comm_ln)
    compare_erg_ln, = ax15.plot(time_data, overall_erg_data[i], c=e_colors[i], label="compare_erg({})".format(i))
    compare_erg_lns.append(compare_erg_ln)

# ##########################################################################
# frame = 0
# time_data.append(frame)

# for i in range(num_compar):
#     for j in range(N):
#         traj_data[i][j][0].append(info[i]['agents'][j].x_log[frame][0])
#         traj_data[i][j][1].append(info[i]['agents'][j].x_log[frame][1])
#         traj_lns[i][j].set_data(traj_data[i][j][0], traj_data[i][j][1])
#         print("traj_lns{}_{}: ".format(i, j), traj_lns[i][j])
#         print(traj_data[i][j][0])
#         print(traj_data[i][j][1])
#         plt.show()
#         plt.savefig("test1{}_{}.png".format(i, j))
#         local_erg_data[i][j].append(info[i]['agents'][j].e_log[frame])
#         local_erg_lns[i][j].set_data(time_data, local_erg_data[i][j])
#         print("local_erg_lns{}_{}: ".format(i, j), local_erg_lns[i][j])
#         print(time_data)
#         print(local_erg_data[i][j])
#         plt.show()
#         plt.savefig("test2{}_{}.png".format(i, j))

#     overall_erg_data[i].append(info[i]['overall_e_log'][frame])
#     overall_erg_lns[i].set_data(time_data, overall_erg_data[i])
#     print("overall_erg_lns{}: ".format(i), overall_erg_lns[i])
#     print(time_data)
#     print(overall_erg_data[i])
#     plt.show()
#     plt.savefig("test3{}.png".format(i))

#     weight_data[i] = info[i]['A_log'][frame]
#     weight_lns[i].set_data(weight_data[i])
#     print("weight_lns{}: ".format(i), weight_lns[i])
#     print(weight_data[i])
#     plt.show()
#     plt.savefig("test4{}.png".format(i))

#     comm_data[i].append(info[i]['communication_cost'][frame])
#     comm_lns[i].set_data(time_data, comm_data[i])
#     print("comm_lns{}: ".format(i), comm_lns[i])
#     print(comm_data[i])
#     plt.show()
#     plt.savefig("test5{}.png".format(i))

#     compare_comm_lns[i].set_data(time_data, comm_data[i])
#     print("compare_comm_lns{}: ".format(i), compare_comm_lns[i])
#     plt.show()
#     plt.savefig("test6{}.png".format(i))

#     prevcommi = 0 if frame == 0 else total_comm_data[i][-1]
#     total_comm_data[i].append(info[i]['communication_cost'][frame]+prevcommi)
#     compare_total_comm_lns[i].set_data(time_data, total_comm_data[i])
#     print("compare_total_comm_lns{}: ".format(i), compare_total_comm_lns[i])
#     print(total_comm_data[i])
#     plt.show()
#     plt.savefig("test7{}.png".format(i))

#     compare_erg_lns[i].set_data(time_data, overall_erg_data[i])
#     print("compare_erg_lns{}: ".format(i), compare_erg_lns[i])
#     plt.show()
#     plt.savefig("test8{}.png".format(i))
# plt.show()
# plt.savefig("test9.png")
# ##########################################################################

def unpack(traj_lns, local_erg_lns, overall_erg_lns, weight_lns, comm_lns, 
            compare_comm_lns, compare_total_comm_lns, compare_erg_lns):
    trajectories = []
    local_ergs = []
    
    for i in range(num_compar):
        for j in range(N):
            trajectories.append(traj_lns[i][j])
            local_ergs.append(local_erg_lns[i][j])
    return (*trajectories, *local_ergs, *overall_erg_lns, *weight_lns, *comm_lns, 
             *compare_comm_lns, *compare_total_comm_lns, *compare_erg_lns)
        

def animate2d_init():
    return unpack(traj_lns, local_erg_lns, overall_erg_lns, weight_lns, comm_lns, 
                    compare_comm_lns, compare_total_comm_lns, compare_erg_lns)

def animate2d_from_logs_update(frame):
    time_data.append(frame)

    for i in range(num_compar):
        for j in range(N):
            traj_data[i][j][0].append(info[i]['agents'][j].x_log[frame][0])
            traj_data[i][j][1].append(info[i]['agents'][j].x_log[frame][1])
            traj_lns[i][j].set_data(traj_data[i][j][0], traj_data[i][j][1])

            local_erg_data[i][j].append(info[i]['agents'][j].e_log[frame])
            local_erg_lns[i][j].set_data(time_data, local_erg_data[i][j])

        overall_erg_data[i].append(info[i]['overall_e_log'][frame])
        overall_erg_lns[i].set_data(time_data, overall_erg_data[i])

        weight_data[i] = info[i]['A_log'][frame]
        weight_lns[i].set_data(weight_data[i])

        comm_data[i].append(info[i]['communication_cost'][frame])
        comm_lns[i].set_data(time_data, comm_data[i])

        compare_comm_lns[i].set_data(time_data, comm_data[i])

        prevcommi = 0 if frame == 0 else total_comm_data[i][-1]
        total_comm_data[i].append(info[i]['communication_cost'][frame]+prevcommi)
        compare_total_comm_lns[i].set_data(time_data, total_comm_data[i])

        compare_erg_lns[i].set_data(time_data, overall_erg_data[i])

    print("frame: ", frame)
    plt.savefig("images/{}_{}.png".format(filename, frame))
    return unpack(traj_lns, local_erg_lns, overall_erg_lns, weight_lns, comm_lns, 
                    compare_comm_lns, compare_total_comm_lns, compare_erg_lns)


frames = num_iter

# animate2d_init()
# for frame in range(frames):
#     animate2d_from_logs_update(frame)
#     plt.show()
#     plt.savefig("test_{}.png".format(frame))

FFwriter = animation.writers['ffmpeg']
writer = FFwriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
anime = animation.FuncAnimation(fig, animate2d_from_logs_update, init_func=animate2d_init, 
                            frames=frames, interval=200, blit=True)  

plt.show()

if filename is not None:
    anime.save(filename+".mp4", writer=writer) 
