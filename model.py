from utils.nes import *
from utils.policy import *

class Actor(nn.Module): # decorator
    def __init__(self, state_size, net, wrap_action, action_size, ddpg):
        super().__init__()
        self.state_size = state_size
        self.net = net
        self.wrap_action = wrap_action
        self.algo = DDPG() if ddpg else PPO(action_size)
    def forward(self, goal, state):
        state = torch.cat([goal, state], 1)
        x = self.net(state)
        x = self.wrap_action(x)
        return self.algo(x)

    def sample_noise(self, _):
        self.net.sample_noise(random.randint(0, len(self.net.layers)))
    def remove_noise(self):
        return

class ActorFactory: # proxy
    def __init__(self, layers, wrap_action, action_size, ddpg):
        self.ddpg = ddpg
#        layers[0]=33
        layers[0]+=3 # temporal hack, dont forget to get rid of
        self.state_size = layers[0]
        self.action_size = action_size
        self.wrap_action = wrap_action
        self.factory = NoisyNetFactory(layers)
        layers[0]-=3
    def head(self):
        return Actor(self.state_size, self.factory.head(), self.wrap_action, self.action_size, self.ddpg)

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

def rl_ibounds(layer):
    b = 1. / np.sqrt(layer.weight.data.size(0))
    return (-b, +b)

def initialize_weights(layer):
    if type(layer) not in [nn.Linear, ]:
        return
#    nn.init.kaiming_uniform_(layer.weight) # does not appear working better ...
    nn.init.uniform_(layer.weight.data, *rl_ibounds(layer))

class Critic(nn.Module):
    def __init__(self, n_actors, n_rewards, state_size, action_size, wrap_value, fc1_units=400, fc2_units=300):
        super().__init__()

        state_size += 3
        self.wrap_value = wrap_value

        #  state_size = 64
        self.fca = nn.Linear(action_size * n_actors, state_size)
        self.fcs = nn.Linear(state_size * n_actors, state_size * n_actors)

        self.fc1 = nn.Linear(state_size * (n_actors + 1), fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)#, bias=False)

        self.apply(initialize_weights)

        self.fc3 = nn.Linear(fc2_units, n_rewards)#, bias=False)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3) # seems this works better ? TODO : proper tests!!

    def forward(self, goals, states, actions):
        # process so actions can contribute to Q-function more effectively ( theory .. )
        actions = self.fca(actions)
        states = torch.cat([goals, states], 1)
        states = F.relu(self.fcs(states)) # push states trough as well
# after initial preprocessing let it flow trough main network in combined fashion
        xs = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1(xs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #  return x
        return self.wrap_value(x)
