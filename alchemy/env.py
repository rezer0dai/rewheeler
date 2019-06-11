from __future__ import print_function

import random
import numpy as np

from collections import deque

class Env:
    def __init__(self, bot, task,
            total_envs, n_history, history_features, state_size,
            n_step, send_delta,
            eval_limit, eval_ratio, max_n_episode, eval_delay,
            mcts_random_cap, mcts_rounds,
            ):

        assert n_step <= send_delta, "for GAE, but we apply it in general, we need to send at least n_step samples ( adjust send_delta accordingly! )"

        self.bot = bot
        self.task = task

        # configs
        self.total_envs = total_envs
        self.state_size = state_size // n_history
        self.n_history = n_history
        self.history_features = history_features
        self.send_delta = send_delta
        self.n_step = n_step

        self.max_n_episode = max_n_episode
        self.eval_limit = eval_limit
        self.eval_ratio = eval_ratio
        self.eval_delay = eval_delay
        self.mcts_random_cap = mcts_random_cap
        self.mcts_rounds = mcts_rounds

        # debug state
        self.ep_count = 0
        self.step_count = 0
        self.score = 0
        self.best_max_step = 0
        self.best_min_step = max_n_episode

    def start(self, callback, dbgout):
        finished, test_scores = self.evaluate()
        if finished:
            return test_scores

        scores = [0]
        while True:
            seeds = [random.randint(0, self.mcts_random_cap)] * self.mcts_rounds
            score = self._simulate(seeds, dbgout)

            finished, test_scores = self.evaluate() if 0 == len(scores) % self.eval_delay else (False, None)
            if finished:
                break

            callback(self.bot, self.task, test_scores, score, seeds, len(scores)) # back to user

            scores.append(score)
        return scores

    def evaluate(self):
        succ = 0
        scores = []
        for _ in range(self.eval_limit):
            for data in self._learning_loop(*self._history(), 0, False):
                pass
            scores.append(self.score)
            succ += self.task.goal_met(self.score)
            if not self.task.goal_met(np.mean(scores)):
                return False, scores # fast fail
        if succ > self.eval_limit * self.eval_ratio:
            print("\n environment solved! ", np.mean(scores))
            print(scores)
        return succ > self.eval_limit * self.eval_ratio, scores

    def _learning_loop(self, f_pi, history, seed, learn_mode):
        self.score = 0
        next_state = self.task.reset(seed, learn_mode)
        steps = 0
        while True:
            steps += 1
            if learn_mode and steps > self.max_n_episode:
#            if steps > self.max_n_episode:
                break

            state = next_state.reshape(len(history), -1)
            for i, s in enumerate(state):
                history[i].append(s.flatten())
            state = np.asarray(history).reshape(len(history), -1)

            goal = self.task.goal()

            if learn_mode:
                a_pi, f_pi = self.bot.explore(goal, state, f_pi, self.step_count)
            else:
                a_pi, f_pi = self.bot.exploit(goal, state, f_pi)

            data = self.task.step(a_pi, learn_mode)
            if data is None:
                break

            action, next_state, reward, done, good = data

            yield (a_pi, goal, state, f_pi, action, reward, good)

            self.score += np.mean(reward)
            if sum(done):
                break

        # last state tunning
        state = next_state.reshape(len(history), -1)
        for i, s in enumerate(state):
            history[i].append(s.flatten())
        state = np.asarray(history).reshape(len(history), -1)

        # last dummy state -> for selfplay, no need action, reward, goo
        # we need goal + history + state
        yield (
                np.zeros(a_pi.shape), # will not be used
                self.task.goal(), state, f_pi,
                np.zeros(action.shape), # will not be used
                np.zeros(reward.shape), # will not be used
                np.zeros(good.shape)) # no need to self-play from this state

    def _simulate(self, seeds, dbgout):
        scores = []
        for e, seed in enumerate(seeds):
            self.ep_count += 1

            goals = []
            states = []
            features = []
            actions = []
            probs = []
            rewards = []
            goods = []

            f_pi, history = self._history()
            features += [f_pi] * 1

            last = 0
            for data in self._learning_loop(f_pi, history, seed, True):
                self.step_count += 1

                a_pi, goal, state, f_pi, action, reward, good = data

                prob = self.bot.log_prob(action)

                # here is action instead of a_pi on purpose ~ let user tell us what action he really take!
                actions.append(action)
                probs.append(prob)
                rewards.append(reward)
                goals.append(goal)
                states.append(state)
                goods.append(good)

                temp = self._share_experience(e, len(states), last)
                if temp != last:
                    exp_delta = (self.send_delta + 2*self.n_step) if last else len(goals)
                    self._share_imidiate(
                            goals[-exp_delta:-self.n_step],
                            states[-exp_delta:-self.n_step],
                            features[-exp_delta:-self.n_step],
                            actions[-exp_delta:-self.n_step],
                            probs[-exp_delta:-self.n_step],
                            rewards[-exp_delta:-self.n_step],
                            goods[-exp_delta:-self.n_step])
                last = temp

                features.append(f_pi)

                # debug
                if sum(good): self._print_stats(e, rewards, a_pi)

            self._share_final(
                    goals[last:],
                    states[last:],
                    features[last:-1],
                    actions[last:],
                    probs[last:],
                    rewards[last:],
                    goods[last:])

            scores.append(self.score)
            self.best_max_step = max(self.best_max_step, len(rewards))
            self.best_min_step = min(self.best_min_step, len(rewards))

            dbgout(self.bot, self.score, actions)

        return scores

    def _history(self):
        f_pi = np.zeros([self.total_envs, self.history_features])
        history = [ deque(maxlen=self.n_history) for _ in range(self.total_envs) ]
        for s in np.zeros([self.n_history, self.state_size]):
            for i in range(len(history)):
                history[i].append(s.flatten())
        return f_pi, history

    def _share_experience(self, e, total, last):
        delta = e + total
        if (delta - self.n_step) % self.send_delta:
            return last# dont overlap
        if total < self.n_step * 3:
            return last# not enough data
        return total - 2 * self.n_step

    def _share_final(self,
            goals, states, features, actions, probs, rewards,
            goods):

        if len(goals) < self.n_step:
            return # ok this ep we scatter

        self._share_imidiate(
            goals, states, features, actions, probs, rewards,
            goods, finished=True)

    def _share_imidiate(self,
            goals, states, features, actions, probs, rewards,
            goods, finished=False):
# just wrapper
        self.bot.step(
            goals, states, features, actions, probs, rewards,
            goods, finished)

    def _print_stats(self, e, rewards, action):
        print("\r[{:5d}>{:6d}::{:2d}] steps = {:4d}, max_step = {:3d}/{:3d}, reward={:2f} <action={}>{}".format(
            self.ep_count, self.step_count, e, len(rewards),
            self.best_max_step, self.best_min_step,
            self.score, action.reshape(self.total_envs, -1)[0].flatten(), " "*20), end="")
