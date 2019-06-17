import numpy as np

from threading import Thread

import torch
from torch.multiprocessing import Process, SimpleQueue

from alchemy.brain import Brain
from alchemy.agent import agent_launch, Agent

#@dataclass
class BrainDescription:
    def __init__(self,
            ddpg,
            count, n_groups, Actor, Critic,
            model_path, save, load, delay,
            resample_delay,
            good_reach, replay_buffer, batch_size,
            sync_delta, learning_delay, learning_repeat,
            fresh_frac, optim_epochs,
            replay_cleaning, prob_treshold,
            ppo_eps, dbgout,
            adv_on, adv_boost,
            lr_actor, lr_critic, clip_norm,
            tau_replay_counter, tau_base, tau_final,
            ):
        self.ddpg = ddpg
        self.count = count
        self.n_groups = n_groups
        self.Actor = Actor
        self.Critic = Critic
        self.model_path = model_path
        self.save = save
        self.load = load
        self.delay = delay
        self.resample_delay = resample_delay
        self.good_reach = good_reach
        self.replay_buffer=replay_buffer
        self.sync_delta=sync_delta
        self.learning_delay=learning_delay
        self.prob_treshold=prob_treshold
        self.learning_repeat=learning_repeat
        self.batch_size=batch_size
        self.replay_cleaning=replay_cleaning
        self.ppo_eps=ppo_eps
        self.dbgout=dbgout
        self.adv_on=adv_on
        self.adv_boost=adv_boost
        self.fresh_frac=fresh_frac
        self.optim_epochs=optim_epochs
        self.tau_replay_counter=tau_replay_counter
        self.tau_base=tau_base
        self.tau_final=tau_final
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.clip_norm=clip_norm

class Bot:
    def __init__(self,
 # brain configs
            encoder,
            brain_descriptions,
            n_actors, n_critics,
            n_history, state_size, action_size,
            n_step, floating_step, gamma,
 # agent configs
            update_goal,
            her_max_ratio,
            gae, gae_tau,
            freeze_delta, freeze_count,
            ):

        self.gates = []
        self.agents = []
        self.brains = []
        self.offsets = [0]
        self.total = 0

        encoder.share_memory()
        for bd in brain_descriptions:
            self.total += bd.count
            self.offsets.append(self.offsets[-1] + bd.count)
            self.gates.append(SimpleQueue())
            self.brains.append(Brain(
                xid=len(self.brains),
                ddpg=bd.ddpg,
                Actor=bd.Actor, Critic=bd.Critic, encoder=encoder, master=(not len(self.brains)),
                n_actors=n_actors, n_critics=n_critics,
                n_history=n_history, state_size=state_size, action_size=action_size,
                resample_delay=bd.resample_delay,
                lr_actor=bd.lr_actor, lr_critic=bd.lr_critic, clip_norm=bd.clip_norm,
                n_step=n_step, gamma=gamma, gae=gae,
                ppo_eps=bd.ppo_eps, dbgout=bd.dbgout,
                adv_on=bd.adv_on, adv_boost=bd.adv_boost,
                model_path=bd.model_path, save=bd.save, load=bd.load, delay=bd.delay,
                ))
            self.brains[-1].share_memory() # make it explicit

            self.agents.append(
                Agent(self.brains[-1],
                    replay_buffer=bd.replay_buffer, update_goal=update_goal,
                    n_groups=bd.n_groups,
                    n_step=n_step, floating_step=floating_step, gamma=gamma, good_reach=bd.good_reach,
                    sync_delta=bd.sync_delta, learning_delay=bd.learning_delay, learning_repeat=bd.learning_repeat, batch_size=bd.batch_size,
                    fresh_frac=bd.fresh_frac, optim_epochs=bd.optim_epochs,
                    replay_cleaning=bd.replay_cleaning, prob_treshold=bd.prob_treshold,
                    her_max_ratio=her_max_ratio,
                    gae=gae, gae_tau=gae_tau,
                    tau_replay_counter=bd.tau_replay_counter, tau_base=bd.tau_base, tau_final=bd.tau_final,
                    freeze_delta=freeze_delta, freeze_count=freeze_count,
                    ))
            continue

            self.agents.append(Process(
                    target=agent_launch,
                    args=(self.gates[-1], self.brains[-1],
                        bd.replay_buffer, update_goal,
                        bd.n_groups,
                        n_step, floating_step, gamma, bd.good_reach,
                        bd.sync_delta, bd.learning_delay, bd.learning_repeat, bd.batch_size,
                        bd.fresh_frac, bd.optim_epochs,
                        bd.replay_cleaning, bd.prob_treshold,
                        her_max_ratio,
                        gae, gae_tau,
                        bd.tau_replay_counter, bd.tau_base, bd.tau_final,
                        freeze_delta, freeze_count,
                        )))

            self.agents[-1].start()

    def __del__(self):
        return
        for gate in self.gates:
            gate.put(None)
        for agent in self.agents:
            agent.join()

    def sync_target(self, a, b, blacklist):
        self.brains[a].sync_target(b, blacklist)
    def sync_explorer(self, a, b, blacklist):
        self.brains[a].sync_explorer(b, blacklist)

    #  def sync_master(self, i, blacklist):
    #      if not i or i >= len(self.brains):
    #          return
    #      self.brains[i].sync_master(blacklist)

    def exploit(self, goal, state, history):
        self.dist = []
        a_pi, f_pi = zip(*self.brains[0].exploit(goal, state, history))
        a_pi, f_pi = np.vstack(a_pi), np.vstack(f_pi)
        return a_pi, f_pi

    def explore(self, goal, state, history, t):
        self.dist = []

        a_pi, f_pi = [], []
        for i, (brain, off) in enumerate(zip(self.brains, self.offsets)):
            indices = range(off, self.offsets[i+1])
            for dist, feat in brain.explore(
                    goal[indices, :],
                    state[indices, :],
                    history[indices, :], t):

                self.dist.append(dist)

                f_pi.append(feat)
                a_pi.append(dist.sample().cpu().numpy())

        a_pi, f_pi = np.vstack(a_pi), np.vstack(f_pi)
        return a_pi, f_pi

    def log_prob(self, actions):
        with torch.no_grad():
            actions = torch.tensor(actions)#.view(len(actions), self.total, -1)

            assert len(self.dist) == len(actions.view(-1, actions.shape[-1])), "...{} {}".format(
                    len(self.dist), actions.view(-1, actions.shape[-1]).shape)

            return torch.cat([
                dist.log_prob(a.view(1, -1)) for dist, a in zip(self.dist, actions.view(-1, actions.shape[-1]))
                ]).view(actions.shape).cpu().numpy()

    def step(self,
            goals, states, features, actions, probs, rewards,
            goods, finished):

        num = len(states) if len(np.shape(states)) > 2 else 1

        reshape = lambda data: np.asarray(data).reshape(
                num, self.total, -1)

        ( goals, states, features, actions, probs, goods ) = (
            reshape(goals), reshape(states), reshape(features),
            reshape(actions), reshape(probs), reshape(goods) )

        rewards = np.asarray(rewards)

        for i, (gate, off) in enumerate(zip(self.gates, self.offsets)):
            indices = range(off, self.offsets[i+1])
            extract = lambda data: data[:, indices, :].reshape(len(data), -1)

            #  gate.put([finished, (
            self.agents[i].train([finished, (
                extract(goals), extract(states), extract(features),
                extract(actions), extract(probs), extract(rewards),
                extract(goods),)])
