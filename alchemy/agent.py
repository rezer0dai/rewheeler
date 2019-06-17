import random
import numpy as np

from utils import policy
from utils.learning_rate import LinearAutoSchedule as LinearSchedule

def agent_launch(
        gate,
        brain,
        replay_buffer, update_goal,
        n_groups,
        n_step, floating_step, gamma, good_reach,
        sync_delta, learning_delay, learning_repeat, batch_size,
        fresh_frac, optim_epochs,
        replay_cleaning, prob_treshold,
        her_max_ratio,
        gae, gae_tau,
        tau_replay_counter, tau_base, tau_final,
        freeze_delta, freeze_count
        ):
    agent = Agent(
        brain,
        replay_buffer, update_goal,
        n_groups,
        n_step, floating_step, gamma, good_reach,
        sync_delta, learning_delay, learning_repeat, batch_size,
        fresh_frac, optim_epochs,
        replay_cleaning, prob_treshold,
        her_max_ratio,
        gae, gae_tau,
        tau_replay_counter, tau_base, tau_final,
        freeze_delta, freeze_count,
        )
    agent.training_loop(gate)

class Agent:
    def __init__(self,
            brain,
            replay_buffer, update_goal, # HER / RNN / GAE essential for ReplayBuffer
            n_groups,
            n_step, floating_step, gamma, good_reach,
            sync_delta, learning_delay, learning_repeat, batch_size, # immidiate learning
            fresh_frac, optim_epochs,
            replay_cleaning, prob_treshold, # PPO
            her_max_ratio,
            gae, gae_tau,
            tau_replay_counter, tau_base, tau_final,
            freeze_delta, freeze_count,
            ):
        # TODO : add to config ...
        self.fresh_frac = fresh_frac
        self.optim_epochs = optim_epochs
        self.floating_step = floating_step

        self.brain = brain
        self.update_goal = update_goal

# encoders freezing!
        self.freeze_delta = freeze_delta
        self.freeze_count = freeze_count
        self.iter = 0
        self.freezed = 0

        self.n_groups = n_groups

        self.n_step = n_step
        self.gamma = gamma
        self.good_reach = good_reach
        self.learning_delay = learning_delay
        self.sync_delta = sync_delta
        self.learning_repeat = learning_repeat
        self.batch_size = batch_size
        self.replay_cleaning = replay_cleaning
        self.prob_treshold = prob_treshold
        self.her_max_ratio = her_max_ratio
        self.gae = gae
        self.gae_tau = gae_tau

        self.replay_buffer = replay_buffer

        self.tau = LinearSchedule(tau_replay_counter, initial_p=tau_base, final_p=tau_final)

        self.counter = 0
        self.reset()

        self.steps = 0
        self.last_train_cap = self.learning_delay

    def reset(self):
        ( self.goals, self.states, self.features, self.actions, self.probs, self.rewards,
          self.n_goals, self.n_states, self.n_features, self.credits, self.discounts, self.goods,
          self.n_steps ) = (
          [], [], [], [], [], [], [], [], [], [], [], [], [] )

    def training_loop(self, share_gate):
        while True:
            exp = share_gate.get()
            if None == exp:
                break
            self.train(exp)

    def train(self, exp):
        finished, exp = exp
        if not finished:
            self._inject(False, exp)
        else:
            self._finish(exp)

    def _try_learn(self):
        if self.steps < self.last_train_cap:
            return

        self.last_train_cap += self.learning_delay

        for batch in self._do_sampling():
            self.counter += 1
            for _ in range(self.optim_epochs): self.brain.learn(batch,
                    self.tau.value() if (0 == self.counter % self.sync_delta) else 0)

    def _finish(self, exp):
        self._inject(True, exp)
        self._update_memory()
        self.reset()

    def _regroup_for_memory(self):
        g = lambda data: data.reshape(len(data), self.n_groups, -1)
        return ( g(self.goals), g(self.states), g(self.features), g(self.actions), g(self.probs), g(self.rewards),
            g(self.n_goals), g(self.n_states), g(self.n_features), g(self.credits), self.discounts.reshape(-1, 1), g(self.goods) )

    def _update_memory(self):
        batch = self._regroup_for_replay()
        prios = self.brain.novelity(batch)

        assert len(self.n_steps) == len(self.discounts) - self.n_step
        g, s, f, a, p, r, n_g, n_s, n_f, n_r, n_d, good = self._regroup_for_memory()
        prios = np.reshape(prios, [len(g), self.n_groups, -1]).mean(2) # per group
        for i in range(g.shape[1]):
            self.replay_buffer.add(
                map(lambda j: (
                    g[j, i], s[j, i], f[j, i], a[j, i], p[j, i], r[j, i],
                    n_g[j, i], n_s[j, i], n_f[j, i], n_r[j, i], n_d[j],
                    self.n_steps[j] if j < len(n_d)-self.n_step else None,
                    bool((j < len(s) - self.n_step) and good[j:j+self.good_reach, i].sum())
                    ), range(len(s))),
                prios[:, i], hash(s[:i:].tostring()))

    def _population(self, batch, limit):
        return random.sample(range(len(batch)), random.randint(
            1, min(limit, len(batch) - 1)))

    def _regroup_for_replay(self):
        g = lambda data: np.reshape(data, [len(data) * self.n_groups, -1])
        return ( g(self.goals), g(self.states), g(self.features), g(self.actions), g(self.probs),
            g(self.n_goals), g(self.n_states), g(self.n_features), g(self.credits), self.discounts.reshape(-1, 1) )

    def _do_sampling(self):
        if max(len(self.replay_buffer), len(self.states)) < self.batch_size:
            return None

        if not len(self.states):
            return None

        batch = np.vstack(zip(*self._regroup_for_replay()))

        for i in range(self.learning_repeat):
            self._encoder_freeze_schedule()

            samples = self._select()
            # keep an eye on latest experience
            population = [] if (not self.fresh_frac or i % self.fresh_frac) else self._population(batch, self.batch_size // self.fresh_frac)
            mini_batch = batch[population] if None == samples else np.vstack([samples, batch[population]])
            yield mini_batch.T

    def _encoder_freeze_schedule(self):
        if not self.freeze_delta:
            return
        self.iter += (0 == self.freezed)
        if self.iter % self.freeze_delta:
            return
        if not self.freezed:
            self.brain.freeze_encoders()
        self.freezed += 1
        if self.freezed <= self.freeze_count:
            return
        self.freezed = 0
        self.brain.unfreeze_encoders()

    def _select(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size, self)
        if None == batch:
            return None

        goals, states, features, actions, probs, _, n_goals, n_states, n_features, credits, discounts, _, _ = batch
        batch = (goals, states, features, actions, probs, n_goals, n_states, n_features, credits, discounts)

        self._update_replay_prios(batch)

        return np.vstack(zip(*batch))

    def _update_replay_prios(self, batch):
        # maybe better to do sum(1) but then we need to propagate this when adding to replay buffer as well ( aka .sum(2).mean(2) )
        prios = self.brain.novelity(batch).mean(1) # mean trough rewards for objectives
# seems we are bit too far for PG ( PPO ) to do something good, replay buffer should abandon those
        if self.replay_cleaning:
            ( states, actions, goals ) = (
                    np.vstack(batch[1]), np.vstack(batch[3]), np.vstack(batch[0]) )

            new_probs = self.brain.reevaluate(goals, states, actions)
            prios[self.prob_treshold[1] < np.abs(new_probs.mean(-1))] = 1e-10
            prios[self.prob_treshold[0] > np.abs(new_probs.mean(-1))] = 1e-10

        self.replay_buffer.update(prios)

    def _assign_credit(self, n_groups, n_steps, rewards, goals, states, features, actions, stochastic):
        regroup = lambda data: np.reshape(data, [ len(data), n_groups, -1 ])

        grouped_rewards = regroup(rewards)
        if not self.gae:
            return policy.k_discount(n_steps, self.gamma), np.concatenate([
                policy.k_step(
                    n_steps,
                    grouped_rewards[:, i], self.gamma
                    ) for i in range(n_groups)], 1)

        grouped_goals = regroup(goals)
        grouped_states = regroup(states)
        grouped_features = regroup(features)
        grouped_actions = regroup(actions)

        return policy.gae_discount(n_steps, self.gamma, self.gae_tau), np.concatenate([
            policy.gae(
                n_steps, grouped_rewards[:len(grouped_states) - 1, i],
                self.brain.qa_future(
                    grouped_goals[:, i], grouped_states[:, i], grouped_features[:, i], grouped_actions[:, i]),
                self.gamma, self.gae_tau
                ) for i in range(n_groups)], 1)

    def _random_n_step_her(self, length, inds):
        do_n_step = lambda n: self.n_step if not self.floating_step else random.randint(n, self.n_step)
        n_step = lambda i: 1 if inds[i] else do_n_step(1 if length-1>i+self.n_step else (length-i-1))
        return self._do_random_n_step(length, n_step)

    def _random_n_step(self, length):
        do_n_step = lambda n: self.n_step if not self.floating_step else random.randint(n, self.n_step)
        n_step = lambda i: do_n_step(1 if length-1>i+self.n_step else (length-i-1))
        return self._do_random_n_step(length, n_step)

    def _do_random_n_step(self, length, n_step):
        n_steps = [ n_step(i) for i in range(length - self.n_step) ]
        indices = np.asarray(n_steps) + np.arange(len(n_steps))
        indices = np.hstack([indices, self.n_step*[-1]])
        return n_steps, indices

    def _redistribute_rewards(self, n_groups, n_steps, indices, rewards, goals, states, features, actions, stochastic):
        # n-step, n-discount, n-return - Q(last state)
        discounts, credits = self._assign_credit(
                n_groups, n_steps, rewards, goals, states, features, actions, stochastic)

        discounts = np.hstack([discounts, self.n_step*[0]])
        credits = np.vstack([credits, np.zeros([self.n_step, len(credits[0])])])

        return ( # for self-play
                credits, discounts,
                goals[indices], states[indices], features[indices],
                )

    def _inject(self, finished, exp):
        goals, states, features, actions, probs, rewards, goods = exp
        if not len(states):
            return # can happen at the end of episode, we just handle it as a notification

        self.steps += len(states) - (self.n_step if not finished else 1)

        n_steps, n_indices = *self._random_n_step(len(rewards)),
        credits, discounts, n_goals, n_states, n_features = self._redistribute_rewards(
                self.n_groups, n_steps, n_indices,
                rewards, goals, states, features, actions, stochastic=True)

        if not finished: # scatter overlaping info
            goods = goods[:-self.n_step]
            goals = goals[:-self.n_step]
            states = states[:-self.n_step]
            features = features[:-self.n_step]
            actions = actions[:-self.n_step]
            probs = probs[:-self.n_step]
            rewards = rewards[:-self.n_step]
            n_goals = n_goals[:-self.n_step]
            n_states = n_states[:-self.n_step]
            n_features = n_features[:-self.n_step]
            credits = credits[:-self.n_step]
            discounts = discounts[:-self.n_step]
            # this branch was not properly evaluated imho ...

        self.goods = np.vstack([self.goods, goods]) if len(self.goods) else goods
        self.goals = np.vstack([self.goals, goals]) if len(self.goals) else goals
        self.states = np.vstack([self.states, states]) if len(self.states) else states
        self.features = np.vstack([self.features, features]) if len(self.features) else features
        self.actions = np.vstack([self.actions, actions]) if len(self.actions) else actions
        self.probs = np.vstack([self.probs, probs]) if len(self.probs) else probs
        self.rewards = np.vstack([self.rewards, rewards]) if len(self.rewards) else rewards
        self.n_goals = np.vstack([self.n_goals, n_goals]) if len(self.n_goals) else n_goals
        self.n_states = np.vstack([self.n_states, n_states]) if len(self.n_states) else n_states
        self.n_features = np.vstack([self.n_features, n_features]) if len(self.n_features) else n_features
        self.credits = np.vstack([self.credits, credits]) if len(self.credits) else credits
        self.discounts = np.hstack([self.discounts, discounts]) if len(self.discounts) else discounts
        self.n_steps = np.hstack([self.n_steps, n_steps]) if len(self.n_steps) else n_steps

        self._try_learn()

    def _her(self, inds):
        collision_free = lambda i, ind: ind-self.n_step>inds[i-1] and ind+1!=inds[i+1]
        hers = [-1] + [ i for i, ind in enumerate(inds[1:-1]) if collision_free(i+1, ind) ]

        pivot = 1
        indices = [ inds[0] ]
        hers.append(len(inds)-1)
        for i, ind in enumerate(inds[1:]):
            if i == hers[pivot] or indices[-1]+1==ind or (0 != random.randint(0, 1 + (i - hers[pivot-1]) // self.her_max_ratio) and indices[-1] == inds[i]):
                indices.append(ind)
            if i == hers[pivot]:
                pivot += 1
        return indices

        non_her = list(filter(lambda i: i not in indices, inds))
        for i in non_her:
            for j in indices:
                assert i+self.n_step < j or i-1 > j, "OPLA {} || {} >< {}\n>{} ({}::{})".format(indices, inds, non_her, hers, i, j)

        return indices

# well TODO : refactor accessing replay buffer entry, from indexing and stacking to proper class and querying
# TODO-2 : common those nasty recalc ifs... => make two separate functions!!
    def reanalyze_experience(self, episode, indices, recalc, cross_experience):
        f, a, p, c, d, n_steps = zip(*[
            [e[0][2], e[0][3], e[0][4], e[0][9], e[0][10], e[0][11] ] for e in episode ])

        inds = sorted(indices)
        cache = np.zeros(len(episode))
        if not recalc: cache[ self._her(inds) ] = 1

        n_steps, n_indices = self._random_n_step_her(len(episode), cache)

# even if we dont recalc here, HER or another REWARD 'shaper' will do its job!!
        r, g, s, n_g, n_s, active = zip(*self.update_goal( # this will change goals not states, OK for Q-function in GAE
            *zip(*[( # magic *
                e[0][5], # rewards .. just so he can forward it to us back
                e[0][0], # goals ..
                e[0][1], # states ..
                episode[n_indices[i]][0][0], # n_goals ..
                episode[n_indices[i]][0][1], # n_states ..
#                e[0][2], # action .. well for now no need, however some emulator may need them
                bool(cache[i]),
                n_steps[i] if len(episode) - self.n_step > i else None,
                ) for i, e in enumerate(episode)])))

        g, s, a, p = np.asarray(g), np.asarray(s), np.asarray(a), np.asarray(p)
        if recalc:
            f = self.brain.recalc_feats(g, s, a)
        else:
            f = np.asarray(f)
        # TODO : appareantelly redistribute rewards functions should be decomposed
        c, d, _, _, _ = self._redistribute_rewards( # here update_goal worked on its own forn of n_goal, so dont touch it here!
                1, n_steps, n_indices, r, g, s, f, a, stochastic=True)

        for i in range(len(episode)):#indices:
            yield ( [ g[i], s[i], f[i], a[i], p[i], r[i],
                    n_g[i], n_s[i], f[n_indices[i]] if len(episode) - self.n_step > i else None,#episode[i][0][8],
                    c[i], d[i], n_steps[i] if len(episode) - self.n_step > i else None,
                    episode[i][0][-1] ], active[i] and episode[i][0][-1] )
