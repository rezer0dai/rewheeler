import numpy as np
import random, copy, sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import policy

torch.set_default_tensor_type('torch.DoubleTensor')

PRIO_ALPHA = .8
PRIO_BETA = .9
PRIO_MIN = 1e-10
PRIO_MAX = 1e2

device = torch.device("cpu")#cuda" if torch.cuda.is_available() else "cpu")

from utils.ac import *
from utils.softnet import SoftUpdateNetwork

class Brain(SoftUpdateNetwork):
    def __init__(self,
            xid, ddpg,
            Actor, Critic, encoder, master,
            n_actors, n_critics,
            n_history, state_size, action_size,
            resample_delay,
            lr_actor, lr_critic, clip_norm,
            n_step, gamma, gae,
            ppo_eps, dbgout,
            adv_on, adv_boost,
            model_path, save, load, delay
            ):

        super().__init__(model_path, save, load, delay)
        self.xid = xid

        self.ddpg = ddpg

        self.state_size = state_size * n_history
        self.action_size = action_size
        self.clip_norm = clip_norm
        self.resample_delay = resample_delay

        self.ppo_eps = ppo_eps
        self.dbgout_ratio = dbgout
        self.adv_on = adv_on
        self.adv_boost = adv_boost

        self.qa = None
        self.td_error = None

        self.count = 0
        self.losses = []

        nes = Actor()
        self.ac_explorer = ActorCritic(encoder, master, n_history,
                    [ nes.head() for _ in range(n_actors) ],
                    [ Critic() for _ in range(n_critics) ])

        self.ac_target = ActorCritic(encoder, master, n_history,
                    [ Actor().head() ],
                    [ Critic() for _ in range(n_critics) ])

        # sync
        for explorer in self.ac_explorer.actor:
            self._soft_update(self.ac_target.actor[0].parameters(), explorer, 1.)
        for i in range(n_critics):
            self._soft_update(self.ac_target.critic[i].parameters(), self.ac_explorer.critic[i], 1.)

        # set optimizers, RMSprop is also good choice !
        self.actor_optimizer = optim.Adam(self.ac_explorer.actor_parameters(), lr=lr_actor)
        self.critic_optimizer = [ optim.Adam(
            self.ac_explorer.critic_parameters(i), lr=lr_critic) for i in range(n_critics) ]

        self.load_models(self.xid, "eac")
        self.save_models_ex(self.xid, "eac")

    def resample(self, t):
        if 0 != t % self.resample_delay:
            return
        for actor in self.ac_explorer.actor:
            actor.sample_noise(t // self.resample_delay)

    def explore(self, goal, state, memory, t): # exploration action
        self.resample(t)
        with torch.no_grad():
            for i, (g, s, h) in enumerate(zip(goal, state, memory)):
                dist, memory_ = self.ac_explorer.act(g, s, h, i)
                yield (
                        dist,
                        memory_.cpu().numpy()
                        )

    def exploit(self, goal, state, memory): # exploitation action
        with torch.no_grad():
            for g, s, h in zip(goal, state, memory):
                dist, memory_ = self.ac_target.act(g, s, h, 0)
                yield (
                        dist.sample().cpu().numpy(),
                        memory_.cpu().numpy()
                        )

    def qa_future(self, goals, states, memory, actions):
        with torch.no_grad():
            actions = torch.tensor(actions)
            return self.ac_target.value(goals, states, memory, actions, 0).cpu().numpy()

    def _backprop(self, optim, loss, params):
        # learn
        optim.zero_grad() # scatter previous optimizer leftovers
        loss.backward() # propagate gradients
        torch.nn.utils.clip_grad_norm_(params, self.clip_norm) # avoid (inf, nan) stuffs
        optim.step() # backprop trigger

    def learn(self, batch, tau):
        """
        Order of learning : Critic first or Actor first
        - depends on who should learn Encoder layer
        - second in turn will learn also encoder, but need to be reflected in ac.py
        - aka *_parameters() should returns also encoder.parameters()
        - currently actor will learn encoder
        """
        goals, states, memory, actions, probs, n_goals, n_states, n_history, n_rewards, n_discounts = batch
        if not len(goals):
            return
        ( goals, states, memory, actions, probs, n_goals, n_states, n_history, n_rewards, n_discounts ) = (
            np.vstack(goals), np.vstack(states), np.vstack(memory), np.vstack(actions), np.vstack(probs),
            np.vstack(n_goals), np.vstack(n_states), np.vstack(n_history), np.vstack(n_rewards), np.vstack(n_discounts) )

        probs = torch.tensor(probs).view(len(probs), -1, self.action_size)
        actions = torch.tensor(actions).view(probs.shape)

        self.losses.append([])

        # func approximators; self play
        with torch.no_grad():
            n_qa, _ = self.ac_target(n_goals, n_states, n_history)
        # TD(0) with k-step estimators
        td_targets = torch.tensor(n_rewards) + torch.tensor(n_discounts) * n_qa
        for i in range(len(self.ac_explorer.critic)):
# learn ACTOR
            # func approximators; self play
            qa, dists = self.ac_explorer(goals, states, memory, i)
            # DDPG + advantage learning with TD-learning
            #td_error = qa - td_targets # w.r.t to self-played action from *0-step* state !!
            td_error = policy.qa_error(qa, td_targets,
                    ddpg=self.ddpg,
                    advantages_enabled=self.adv_on,
                    advantages_boost=self.adv_boost)

            if not self.ddpg: td_error = policy.policy_loss(
                    probs.mean(2),
                    torch.cat([
                        dist.log_prob(actions[:, j].view(len(actions), -1)) for j, dist in enumerate(dists)
                        ], 1).view(probs.shape).mean(2),
                    td_error,
                    self.ppo_eps, self.dbgout_ratio)

            td_error = td_error.sum(1) # we try to maximize its sum/mean as cooperation matters now
            actor_loss = -td_error.mean()
            # learn!
            self._backprop(self.actor_optimizer, actor_loss, self.ac_explorer.actor_parameters())

# learn CRITIC
            # estimate reward
            q_replay = self.ac_explorer.value(goals, states, memory, actions, i)
            # calculate loss via TD-learning
            critic_loss = F.mse_loss(q_replay, td_targets)#F.smooth_l1_loss(q_replay, td_targets)#
            # learn!
            self._backprop(self.critic_optimizer[i], critic_loss, self.ac_explorer.critic_parameters(i))
            # propagate updates to target network ( network we trying to effectively learn )
            self._soft_update(self.ac_explorer.critic[i].parameters(), self.ac_target.critic[i], tau)

            self.losses[-1].append(critic_loss.item())

        # propagate updates to target network ( network we trying to effectively learn )
        self._soft_update_mean(self.ac_explorer.actor, self.ac_target.actor[0], tau)

        self.save_models(self.xid, "eac")

        for explorer in self.ac_explorer.actor:
            explorer.remove_noise() # lets go from zero ( noise ) !!

        self.losses[-1] = [actor_loss.item()]+self.losses[-1]
        return td_error.detach().cpu().numpy(),

    def get_losses(self):
        return self.losses

    def novelity(self, batch):
        goals, states, memory, actions, probs, n_goals, n_states, n_history, n_rewards, n_discounts = batch

        ( goals, states, memory, actions, n_goals, n_states, n_history, n_rewards, n_discounts ) = (
            np.vstack(goals), np.vstack(states), np.vstack(memory), np.vstack(actions),
            np.vstack(n_goals), np.vstack(n_states), np.vstack(n_history), np.vstack(n_rewards), np.vstack(n_discounts) )

        n_discounts = np.vstack([ n_discounts ] * (len(n_states) // len(n_discounts)))
        with torch.no_grad():
            n_qa, _ = self.ac_target(n_goals, n_states, n_history)
            td_targets = torch.tensor(n_rewards) + torch.tensor(n_discounts) * n_qa
            self.qa, _ = self.ac_explorer(goals, states, memory, 0)
            self.td_error = self.qa - td_targets

        return self.td_error.abs().cpu().numpy()

    def recalc_feats(self, goals, states, actions):
        with torch.no_grad():
            states = torch.tensor(states).view(len(states), -1, self.state_size)
            _, f = zip(*[
                    self.ac_target.encoder.extract_features(
                        states[:, i]) for i in range(states.size(1)) # encoder is shared
                    ])
        return np.hstack(f)

    def reevaluate(self, goals, states, actions):
        with torch.no_grad():
            states = torch.tensor(states).view(len(states), -1, self.state_size)
            s, f = zip(*[
                    self.ac_target.encoder.extract_features(
                        states[:, i]) for i in range(states.size(1)) # encoder is shared
                    ])
            s = torch.cat(s, 1)

            regroup = lambda data: data.view(len(goals), states.size(1), -1)
            ( s, a, g ) = (
                    regroup(s), regroup(torch.tensor(actions)), regroup(torch.tensor(goals)) )

            p = np.concatenate([
                    self.ac_target.actor[0](
                    #  self.ac_explorer.actor[i % len(self.ac_explorer.actor)](
                        g[:, i], s[:, i]).log_prob( # this should be ac_explorer, but need to enum states
                        a[:, i]).cpu().numpy() for i in range(s.size(1)) ], 1)
# on the other side beiing target is maybe worth to try!!
# diff is for PPO -> now we have probability ratio combining target and latest explorer, which sounds okish ? :)
        return p

    def freeze_encoders(self):
        self.ac_explorer.freeze_encoders()
        self.ac_target.freeze_encoders()

    def unfreeze_encoders(self):
        self.ac_explorer.unfreeze_encoders()
        self.ac_target.unfreeze_encoders()
