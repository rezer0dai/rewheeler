import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from utils.encoders import GoalGlobalNorm

class ActorCritic(nn.Module): # share common preprocessing layer!
    # encoder could be : RNN, CNN, RBF, BatchNorm / GlobalNorm, others.. and combination of those
    def __init__(self, encoder, master, n_history, actor, critic):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.master = master

        self.goaler = GoalGlobalNorm(3)

        self.n_history = n_history
        self.encoded_state_size = encoder.out_size()

        for i, a in enumerate(self.actor):
            self.add_module("actor_%i"%i, a)
        for i, c in enumerate(self.critic):
            self.add_module("critic_%i"%i, c)

        self.encoder_grads = [ p.requires_grad for p in self.encoder.parameters()]

    def parameters(self):
        assert False, "should not be accessed!"

# TODO : where to make sense to train encoder -> at Actor, Critic, or both ??
    def actor_parameters(self):
        return np.concatenate([
#            [ p for p in self.encoder.parameters() if p.requires_grad ] if self.master else [],
            np.concatenate([list(actor.parameters()) for actor in self.actor], 0)])

    def critic_parameters(self, ind):
        c_i = ind if ind < len(self.critic) else 0
        return np.concatenate([
            [ p for p in self.encoder.parameters() if p.requires_grad ] if self.master else [],
            list(self.critic[c_i].parameters())])

    def forward(self, goals, states, memory, ind = 0):
        assert ind != -1 or len(self.critic) == 1, "you forgot to specify which critic should be used"

        ( goals, states, memory ) = ( # data format conversion
                torch.from_numpy(goals), torch.from_numpy(states), torch.from_numpy(memory) )

        states, _ = self.encoder(states, memory)
        states = states.view(
                len(goals), -1, self.encoded_state_size)

        goals = self.goaler(goals)
        goals = goals.view(goals.size(0), states.size(1), -1)

        dists = []
        actions = []
        for i in range(states.size(1)):
            a_i = (i % len(self.actor)) if states.size(1) > 1 or 1 == len(self.actor) else random.randint(0, len(self.actor)-1)
            dist = self.actor[a_i](goals[:, i, :], states[:, i, :])
            dists.append(dist)
            pi = dist.sample()
            actions.append(pi)
        actions = torch.cat(actions, 1)

        states = states.view(states.size(0), -1)
        goals = goals.view(states.size(0), 0-1)
        return self.critic[ind](goals, states, actions), dists

    def value(self, goals, states, memory, actions, ind):
        ( goals, states, memory ) = ( # data format conversion
                torch.from_numpy(goals), torch.from_numpy(states), torch.from_numpy(memory) )

        states, _ = self.encoder(states, memory)
        states = states.view(len(memory), -1)
        actions = actions.view(len(actions), -1)
        goals = self.goaler(goals)
        goals = goals.view(states.size(0), -1)
        return self.critic[ind](goals, states, actions)

    def act(self, goals, states, memory, ind):
        ( goals, states, memory ) = ( # data format conversion
                torch.from_numpy(goals), torch.from_numpy(states), torch.from_numpy(memory) )

        ind = ind % len(self.actor)

        states, memory = self.encoder(states, memory)
        states = states.view(1, -1)
        memory = memory.view(1, -1)
        goals = self.goaler(goals)
        goals = goals.view(1, -1)

        pi = self.actor[ind](goals, states)
        return pi, memory

    def freeze_encoders(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        for g, p in zip(self.encoder_grads, self.encoder.parameters()):
            p.requires_grad = g
