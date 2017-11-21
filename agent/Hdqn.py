import torch
from torch.autograd import Variable
from model.neural_network import neural_network
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import torch.nn.functional as F
from collections import namedtuple
#random.seed(132)


default_meta_epsilon = 1
default_epsilon = 1.0
default_actor_epsilon = [1.0]*6
default_tau = 0.001
default_n_samples = 128
default_gamma = 1
default_meta_nodes = [6, 6]
default_nodes = [12, 2]
default_tau = 0.001
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
default_optimizer_spec = OptimizerSpec(
	constructor=optim.RMSprop,
	kwargs=dict(lr=0.00025, alpha=0.95, eps=1e-06),
)

def mse_loss(input, target):
	return torch.sum((input - target)^2) / input.data.nelement()

loss_function = nn.MSELoss()



class Hdqn:

	def __init__(self, meta_epsilon = default_meta_epsilon, epsilon=default_epsilon, actor_epsilon = default_actor_epsilon,
		tau = default_tau, n_samples=default_n_samples, gamma=default_gamma, optimizer_spec = default_optimizer_spec):
		self.meta_epsilon = meta_epsilon
		self.actor_epsilon = actor_epsilon
		self.gamma = default_gamma
		self.tau = default_tau
		self.epsilon = default_epsilon
		self.goal_selected = np.ones(6)
		self.goal_success = np.zeros(6)
		self.n_samples = default_n_samples
		self.memory = []
		self.meta_memory = []
		self.meta_controller = self.meta_controller()
		self.target_meta_controller = self.target_meta_controller()
		self.actor = self.actor()
		self.target_actor = self.target_actor()
		self.target_tau = tau
		self.meta_optimiser = optimizer_spec.constructor(self.meta_controller.parameters(), **optimizer_spec.kwargs)
		self.actor_optimizer = optimizer_spec.constructor(self.actor.parameters(), **optimizer_spec.kwargs)


	def meta_controller(self):
		meta = neural_network(default_meta_nodes)
		return meta

	def target_meta_controller(self):
		meta = neural_network(default_meta_nodes)
		return meta

	def actor(self):
		actor = neural_network(default_nodes)
		return actor

	def target_actor(self):
		actor = neural_network(default_nodes)
		return actor

	def select_move(self, state, goal, goal_value):
		vector = np.concatenate([state, goal], axis=1)
		#print "vector"
		if random.random() < self.actor_epsilon[goal_value-1]:
			#print "Here ------>", self.actor(Variable(torch.from_numpy(vector).float())).data.numpy()
			return self.actor(Variable(torch.from_numpy(vector).type(torch.FloatTensor), volatile=True)).data.max(1)[1]
		return torch.IntTensor([random.randrange(2)])

	def select_goal(self, state):
		if self.meta_epsilon < random.random():
			pred = self.meta_controller(Variable(torch.from_numpy(state).type(torch.FloatTensor), volatile=True))
			#print("pred shape: " + str(pred.shape))
			return pred.data.max(1)[1]
		print("Exploring");
		return torch.IntTensor([random.randrange(6)])

	def criticize(self, goal, next_state):
		return 1.0 if goal == next_state else 0.0

	def store(self, experience, meta=False):
		if meta:
			self.meta_memory.append(experience)
			if len(self.meta_memory) > 1000000:
				self.meta_memory = self.meta_memory[-100:]
		else:
			self.memory.append(experience)
			if len(self.memory) > 1000000:
				self.memory = self.memory[-1000000:]

	def _update(self):
		if len(self.memory) < self.n_samples:
			return
		exps = [random.choice(self.memory) for _ in range(self.n_samples)]
		state_vectors = np.squeeze(np.asarray([np.concatenate([exp.state, exp.goal], axis=1) for exp in exps]))
		
		state_vectors_var = Variable(torch.from_numpy(state_vectors).type(torch.FloatTensor))

		action_batch = np.array([exp.action for exp in exps])
		action_batch_var = Variable(torch.from_numpy(action_batch).long())

		reward_batch = np.array([exp.reward for exp in exps])
		reward_batch_var = Variable(torch.from_numpy(reward_batch).type(torch.FloatTensor))

		done_batch = np.array([exp.reward for exp in exps])
		not_done_batch_mask = Variable(torch.from_numpy(1- done_batch).type(torch.FloatTensor))
		#print "state_vectors", state_vectors
		next_state_vectors = np.squeeze(np.asarray([np.concatenate([exp.next_state, exp.goal], axis=1) for exp in exps]))
		next_state_vectors_var = Variable(torch.from_numpy(next_state_vectors).type(torch.FloatTensor))
		
		try:
			reward_vectors = self.actor(state_vectors_var).gather(1, action_batch_var.unsqueeze(1))
		except Exception as e:
			state_vectors = np.expand_dims(state_vectors, axis=0)
			reward_vectors = self.actor(state_vectors_var).gather(1, action_batch_var.unsqueeze(1))
		
		try:
			next_state_max_reward = self.target_actor(next_state_vectors_var).detach().max(1)[0]
		except Exception as e:
			next_state_vectors = np.expand_dims(next_state_vectors, axis=0)
			next_state_max_reward = self.target_actor(next_state_vectors_var).detach().max(1)[0]
		
		'''
		new_reward_vectors = deepcopy(reward_vectors.data.numpy())
		#print new_reward_vectors, reward_vectors.data.numpy(), "new and old reward_vectors"
		#print "new_rew", new_reward_vectors
		for i, exp in enumerate(exps):
			#print exp
			new_reward_vectors[i][exp.action] = exp.reward
			#print "val", new_reward_vectors[i][exp.action], exp.reward
			if not exp.done:
				new_reward_vectors[i][exp.action] = new_reward_vectors[i][exp.action] + self.gamma * max(next_state_reward_vectors[i])
				#print "val", new_reward_vectors[i][exp.action], self.gamma * max(next_state_reward_vectors[i])
		#reward_vectors = np.asarray(reward_vectors)
		'''
		next_state_reward_vectors = not_done_batch_mask * next_state_max_reward
		target_Q_values = reward_batch_var + ( self.gamma * next_state_reward_vectors)

		loss = F.smooth_l1_loss(reward_vectors, target_Q_values)

		#print"state_dict", self.actor.state_dict()
		#print "new reward_vectors", torch.from_numpy(new_reward_vectors)
		self.target_actor.load_state_dict(self.actor.state_dict())

		
		'''
		actor_weights = self.actor.parameters()
		actor_target_weights = self.target_actor.parameters()
		
		for actor_target_weight, actor_weight in zip(actor_target_weights, actor_weights):
			actor_target_weight.data = self.target_tau * actor_weight.data + (1 - self.target_tau) * actor_target_weight.data
		
		'''
		
		


		#optimizer = optim.RMSprop(self.actor.parameters(), lr=0.001, alpha=0.9, eps=1e-06)

		self.actor_optimizer.zero_grad()

		
		#print new_reward_vectors, "new"
		#print reward_vectors.data.numpy(), "num"
		
		#print "loss", loss
		loss.backward()
		for param in self.actor.parameters():
			param.grad.data.clamp_(-1, 1)
		self.actor_optimizer.step()

		#self.actor.fit(state_vectors, reward_vectors, verbose=0)
		
		#Update target network
		

	def _update_meta(self):
		if len(self.meta_memory) < self.n_samples:
			return
		exps = [random.choice(self.meta_memory) for _ in range(self.n_samples)]
		state_vectors = np.concatenate([exp.state for exp in exps])
		#print "state_vectors", state_vectors
		state_vectors_var = Variable(torch.from_numpy(state_vectors).type(torch.FloatTensor))

		goal_batch = np.array([exp.goal for exp in exps])
		#print "goal_batch", goal_batch
		goal_batch_var = Variable(torch.from_numpy(goal_batch).long())

		reward_batch = np.array([exp.reward for exp in exps])
		reward_batch_var = Variable(torch.from_numpy(reward_batch).type(torch.FloatTensor))

		done_batch = np.array([exp.reward for exp in exps])
		not_done_batch_mask = Variable(torch.from_numpy(1- done_batch).type(torch.FloatTensor))
		#print "state_vectors", state_vectors
		next_state_vectors = np.concatenate([exp.next_state for exp in exps])
		next_state_vectors_var = Variable(torch.from_numpy(next_state_vectors).type(torch.FloatTensor))
		
		try:
			reward_vectors = self.meta_controller(state_vectors_var).gather(1, goal_batch_var.unsqueeze(1))
		except Exception as e:
			state_vectors = np.expand_dims(state_vectors, axis=0)
			reward_vectors = self.meta_controller(state_vectors_var).gather(1, goal_batch_var.unsqueeze(1))
		
		try:
			next_state_max_reward = self.target_meta_controller(next_state_vectors_var).detach().max(1)[0]
		except Exception as e:
			next_state_vectors = np.expand_dims(next_state_vectors, axis=0)
			next_state_max_reward = self.target_meta_controller(next_state_vectors_var).detach().max(1)[0]
		
		'''
		new_reward_vectors = deepcopy(reward_vectors.data.numpy())
		#print new_reward_vectors, reward_vectors.data.numpy(), "new and old reward_vectors"
		#print "new_rew", new_reward_vectors
		for i, exp in enumerate(exps):
			#print exp
			new_reward_vectors[i][exp.action] = exp.reward
			#print "val", new_reward_vectors[i][exp.action], exp.reward
			if not exp.done:
				new_reward_vectors[i][exp.action] = new_reward_vectors[i][exp.action] + self.gamma * max(next_state_reward_vectors[i])
				#print "val", new_reward_vectors[i][exp.action], self.gamma * max(next_state_reward_vectors[i])
		#reward_vectors = np.asarray(reward_vectors)
		'''
		next_state_reward_vectors = not_done_batch_mask * next_state_max_reward
		target_Q_values = reward_batch_var + ( self.gamma * next_state_reward_vectors)

		loss = F.smooth_l1_loss(reward_vectors, target_Q_values)

		#print"state_dict", self.actor.state_dict()
		#print "new reward_vectors", torch.from_numpy(new_reward_vectors)
		self.target_meta_controller.load_state_dict(self.meta_controller.state_dict())

		
		'''
		actor_weights = self.actor.parameters()
		actor_target_weights = self.target_actor.parameters()
		
		for actor_target_weight, actor_weight in zip(actor_target_weights, actor_weights):
			actor_target_weight.data = self.target_tau * actor_weight.data + (1 - self.target_tau) * actor_target_weight.data
		
		'''
		
		
		#optimizer = optim.RMSprop(self.actor.parameters(), lr=0.001, alpha=0.9, eps=1e-06)

		self.meta_optimiser.zero_grad()

		
		#print new_reward_vectors, "new"
		#print reward_vectors.data.numpy(), "num"
		
		#print "loss", loss
		loss.backward()
		for param in self.meta_controller.parameters():
			param.grad.data.clamp_(-1, 1)
		self.meta_optimiser.step()

		#self.actor.fit(state_vectors, reward_vectors, verbose=0)
		
		#Update target network

	def update(self, meta=False):
			if meta:
				self._update_meta()
			else:
				self._update()