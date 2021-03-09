import numpy as np
from scipy.fft import dctn, idctn
from regions.core import PixCoord
from regions import CirclePixelRegion
from scipy.signal import convolve2d
import casadi
import time

import matplotlib.pyplot as plt

class ErgodicController(object):

	def __init__(
		self, env, x_init, 
		min_dist=.3, nfourier=20, t_horizon=2, t_history=0, mode='mi', gui=True
	):
		self.x = x_init
		self.bounds = env.bounds
		self.u_max = env.get_u_max()
		self.min_dist = min_dist
		self.nfourier = nfourier
		self.t_horizon = t_horizon
		self.nagents = x_init.shape[0]
		self.tp_rate = env.tp_rate
		self.fp_rate = env.fp_rate
		self.u = [np.zeros((2, t_horizon)) for i in range(self.nagents)]

		K1, K2 = np.meshgrid(
			np.arange(nfourier), np.arange(nfourier), indexing='ij'
		)
		self.k1 = K1.flatten() * np.pi
		self.k2 = K2.flatten() * np.pi

		self.hk = np.ones(self.k1.shape[0])
		self.hk[self.k1 != 0] *= np.sqrt(.5)
		self.hk[self.k2 != 0] *= np.sqrt(.5)

		s = (2 + 1) / 2
		self.Lambdak = (1 + np.square(self.k1) + np.square(self.k2))**(-s)

		self.t_history = t_history
		self.ck_history_cum = []

		r = env.target_r + env.view_r
		npix = int(np.ceil(r/ env.grid_dx) * 2 + 1)
		reg = CirclePixelRegion(
			center=PixCoord(npix, npix), radius=r / env.grid_dx
		)
		self.footprint_mask = reg.to_mask(mode='exact').data

		self.image1 = None
		self.image2 = None
		self.lines = None

		# self.change_paramt0 = 5
		# self.change_param = .01

		self.change_paramt0 = .1
		self.change_param = .01

		self.mode = mode
		self.gui = gui

	def update_controls(self, target_belief, x_obs, x):
		ts = time.time()
		self.x = x
		nx, ny = target_belief.shape
		xmin, xmax, ymin, ymax = self.bounds
		L1 = xmax - xmin
		L2 = ymax - ymin

		if self.mode == 'mi':
			mu = convolve2d(target_belief, self.footprint_mask, mode='same')
			pos = self.tp_rate*mu + self.fp_rate*(1-mu)
			mi = (
			    -self.tp_rate*mu*np.log(pos / self.tp_rate) - 
			    (1-self.tp_rate)*mu*np.log((1-pos) / (1-self.tp_rate)) - 
			    self.fp_rate*(1-mu)*np.log(pos / self.fp_rate) -
			    (1-self.fp_rate)*(1-mu)*np.log((1-pos)/(1-self.fp_rate))
			)
			info = mi / np.sum(mi)
		elif self.mode == 'p':
			mu = convolve2d(target_belief, self.footprint_mask, mode='same')
			info = mu / np.sum(mu)
		elif self.mode == 'b':
			info = target_belief / np.sum(target_belief)
		elif self.mode == 'alpha':
			mu = convolve2d(target_belief, self.footprint_mask, mode='same')
			pos = self.tp_rate*mu + self.fp_rate*(1-mu)
			alpha = .5
			ami = (1 / (1 - alpha)) * (
				pos*np.log(mu*(self.tp_rate/pos)**alpha + (1-mu)*(self.fp_rate/pos)**alpha) +
				(1-pos)*np.log(mu*((1-self.tp_rate)/(1-pos))**alpha + (1-mu)*((1-self.fp_rate)/(1-pos))**alpha)
			)
			info = ami / np.sum(ami)

		muk = dctn(info * np.sqrt(nx) * np.sqrt(ny), type=2, norm='ortho')
		muk = muk[0:self.nfourier, 0:self.nfourier].flatten()
		
		x1_comp = np.cos(np.outer((x_obs[:, 0] - xmin)/L1, self.k1))
		x2_comp = np.cos(np.outer((x_obs[:, 1] - ymin)/L2, self.k2))
		ck_prev = (1 / self.hk) * np.sum(x1_comp * x2_comp, axis=0)

		if len(self.ck_history_cum) == 0:
			self.ck_history_cum.append(ck_prev)
			ck_init = np.zeros(self.k1.shape[0])
			t_total = self.t_horizon
		else:
			prev_cum = self.ck_history_cum[-1]
			self.ck_history_cum.append(prev_cum + ck_prev)
			if len(self.ck_history_cum) > self.t_history:
				ck_init = self.ck_history_cum[-1] - self.ck_history_cum[-self.t_history-1]
				t_total = self.t_history + self.t_horizon
			else:
				ck_init = self.ck_history_cum[-1]
				t_total = len(self.ck_history_cum) + self.t_horizon

		opti = casadi.Opti()
		v_u = [opti.variable(2, self.t_horizon) for i in range(self.nagents)]
		v_x = [casadi.cumsum(v_u[i], 2) + x[i, :] for i in range(self.nagents)]
		v_ck = (ck_init + (1 / self.hk) * sum([
			casadi.sum2(
				casadi.cos(self.k1 @ (v_x[i][0, :] - xmin)/L1) * 
				casadi.cos(self.k2 @ (v_x[i][1, :] - ymin)/L2)
			) for i in range(self.nagents)
		])) / (self.nagents * t_total)

		erg_metric = casadi.sum1(self.Lambdak * (v_ck - muk)**2)
		change_penalties = []
		for i in range(self.nagents):
			opti.subject_to(casadi.sum2(v_u[i]**2) < self.u_max**2)
			opti.subject_to(v_x[i][0, :] > xmin)
			opti.subject_to(v_x[i][0, :] < xmax)
			opti.subject_to(v_x[i][1, :] > ymin)
			opti.subject_to(v_x[i][1, :] < ymax)
			for j in range(i + 1, self.nagents):
				opti.subject_to(casadi.sum1((v_x[i] - v_x[j])**2) > self.min_dist**2)

			u_init = np.zeros((2, self.t_horizon))
			u_init[:, 0:-1] = self.u[i][:, 1:]

			opti.set_initial(v_u[i], u_init)
			change_penalties.append(self.change_paramt0 * casadi.sum1(casadi.sum2((v_u[i][:, 0] - self.u[i][:, 0])**2)))
			change_penalties.append(self.change_param * casadi.sum1(casadi.sum2((v_u[i][:, 1:] - v_u[i][:, 0:-1])**2)))

		change_cost = sum(change_penalties)
		opti.minimize(erg_metric + change_cost)

		p_opts = {}
		# s_opts = {'max_cpu_time': .09, 'print_level': 0}
		s_opts = {'print_level': 0}
		opti.solver('ipopt', p_opts, s_opts)
		sol = opti.solve()
		self.u = [sol.value(v_u[i]) for i in range(self.nagents)]
		xs = [sol.value(v_x[i]) for i in range(self.nagents)]

		# try:
		# 	sol = opti.solve()
		# 	self.u = [sol.value(v_u[i]) for i in range(self.nagents)]
		# 	xs = [sol.value(v_x[i]) for i in range(self.nagents)]
		# except:
		# 	self.u = [opti.debug.value(v_u[i]) for i in range(self.nagents)]
		# 	xs = [opti.debug.value(v_x[i]) for i in range(self.nagents)]

		action = np.zeros((self.nagents, 2))
		for i in range(self.nagents):
			action[i, :] = xs[i][:, 0]

		t_comp = time.time() - ts
		self.x = x

		if self.gui:
			xs_plot = [np.insert(xs[i], 0, self.x[i, :], axis=1) for i in range(self.nagents)]
			# display information map, planned trajectories, and 
			if self.image1 is None:
				fig, (ax1, ax2) = plt.subplots(2, 1, num='erg', figsize=(2, 4))
				self.image1 = ax1.imshow(
					info.T / np.max(info), extent=self.bounds, origin='lower', cmap='gray'
				)
				self.image2 = ax2.imshow(
					target_belief.T / np.max(target_belief), extent=self.bounds, 
					origin='lower', cmap='gray'
				)
				ax1.set_title('Information Map')
				ax2.set_title('Target Belief')
				ax1.axis('off')
				ax2.axis('off')
				self.lines = []
				self.inits = []
				for i in range(self.nagents):
					line, = ax1.plot(xs[i][0, :], xs[i][1, :], 'r.-')#, linestyle=':')
					self.lines.append(line)
					init, = ax1.plot(self.x[i, 0], self.x[i, 1], 'g.')
					self.inits.append(init)
			else:
				plt.figure('erg')
				self.image1.set_data(info.T / np.max(info))
				self.image2.set_data(target_belief.T / np.max(target_belief))
				for i in range(self.nagents):
					self.lines[i].set_xdata(xs[i][0, :])
					self.lines[i].set_ydata(xs[i][1, :])
					self.inits[i].set_xdata(self.x[i, 0])
					self.inits[i].set_ydata(self.x[i, 1])

			plt.draw()
			plt.pause(.001)

		return action, t_comp