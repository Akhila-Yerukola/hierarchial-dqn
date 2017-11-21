import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class neural_network(nn.Module):
	def __init__(self, nodes):
		super(neural_network, self).__init__()
		self.fc1 = nn.Linear(nodes[0], 256)
		self.fc2 = nn.Linear(256, nodes[1])

	def forward(self, x):
		x = F.relu(self.fc1(x))
		return self.fc2(x)


