import numpy as np
import random

CLOSE_ENOUGH = 1.15

def extract_goal(state):
    return state[-4-3:-1-3]

# https://github.com/Unity-Technologies/ml-agents/blob/master/UnitySDK/Assets/ML-Agents/Examples/Reacher/Scripts/ReacherAgent.cs
def transform(obs):
    return np.hstack([
        obs[3+4+3+3:3+4+3+3+3], #pendulumB position
        obs[:3+4+3+3], # pendulumA info
        obs[3+4+3+3+3:-4-3], # pundulumB rest of info
        obs[-1-3:] #speed + hand position
        ])

def goal_distance(goal_a, goal_b):
#    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[:3] - goal_b[:3])

def fun_reward(s, n, goal, her): # 3D navigation
    return (
            -.01 * (5 * CLOSE_ENOUGH < np.abs(s[0] - goal[0])),
            -.01 * (5 * CLOSE_ENOUGH < np.abs(s[1] - goal[1])),
            -.01 * (5 * CLOSE_ENOUGH < np.abs(s[2] - goal[2])),

            +.01 * (3 * CLOSE_ENOUGH > np.abs(s[0] - goal[0])),
            +.01 * (3 * CLOSE_ENOUGH > np.abs(s[1] - goal[1])),
            +.01 * (3 * CLOSE_ENOUGH > np.abs(s[2] - goal[2])),

            -.01 * (1 * CLOSE_ENOUGH > np.abs(s[0] - goal[0])),
            -.01 * (1 * CLOSE_ENOUGH > np.abs(s[1] - goal[1])),
            -.01 * (1 * CLOSE_ENOUGH > np.abs(s[2] - goal[2])),
            )

def goal_select(s, trajectory, gid):
    return random.randint(0, len(trajectory)-1)

def update_goal_curyctor(n_step):
    MAX_HER_STEP = 1
    def update_goal(rewards, goals, states, n_goals, n_states, update, n_steps):
        gid = 0
        delta = 0
        for i, (g, s, n_g, n, u, step) in enumerate(zip(goals, states, n_goals, n_states, update, n_steps)):
            her_active = bool(sum(update[(i-MAX_HER_STEP) if MAX_HER_STEP < i else 0:i]))

            if not her_active and u: # only here we do HER approach and setuping new goal
# last n-steps are by design *NOT selected* to replay anyway
                gid = goal_select(s, goals[:-n_step-MAX_HER_STEP], 0)
                delta = 0

            if her_active or u:
                if gid>=0 and gid+delta+n_step<len(goals) and i<len(states)-n_step: # actually HER goal was assigned
                    assert step is not None, "step is none ... {} {} {} {}".format(i, gid, delta, len(states))# 1 11 0 50
                    g, n_g = goals[gid+delta], goals[gid+delta+step]
                delta += 1

            yield (
                fun_reward(s, n, g, True),
                g, s,
                n_g, n,
                gid<0 or gid+delta+MAX_HER_STEP<len(goals)-n_step
            )
    return update_goal

# TEMPORARY IMPLMENTATION ~ testing on Tennis environment from UnityML framework
class Task:
    def __init__(self):
        from unityagents import UnityEnvironment
        self.ENV = UnityEnvironment(file_name='./reach/Reacher.x86_64')
#        self.ENV = UnityEnvironment(file_name='./data/Tennis.x86_64')
        self.BRAIN_NAME = self.ENV.brain_names[0]
        self.random_cut = None

    def reset(self, seed, learn_mode):
#        einfo = self.ENV.reset(config={"goal_size":4.4 * CLOSE_ENOUGH, "goal_speed":.3})[self.BRAIN_NAME]
        einfo = self.ENV.reset()[self.BRAIN_NAME]
        self.random_cut = random.randint(0, len(einfo.vector_observations) - 1)
        states = self._reflow(einfo.vector_observations)
        self._decouple(states)
        return self.states

    def _reflow(self, data):
        return np.vstack([ data[self.random_cut:], data[:self.random_cut] ])
    def _deflow(self, data):
        return np.vstack([ data[-self.random_cut:], data[:-self.random_cut] ])

    def _decouple(self, states):
        self.goals, self.states = zip(
            *[ (extract_goal(s), transform(s)) for s in states ])
        self.goals = np.vstack(self.goals)
        self.states = np.vstack(self.states)

    def goal(self):
        return self.goals#.reshape(len(self.goals), -1)
        return np.zeros([20, 1])
        return np.zeros([2, 1])

    def step(self, actions, learn_mode):
        act_env = self._deflow(actions).reshape(-1)
        einfo = self.ENV.step(act_env)[self.BRAIN_NAME]

        states = self._reflow(einfo.vector_observations)
        dones = self._reflow(np.asarray(einfo.local_done).reshape(len(states), -1))
        rewards = self._reflow(np.asarray(einfo.rewards).reshape(len(states), -1))
        goods = np.ones([len(rewards), 1])

        self._decouple(states)

        if not learn_mode:#True:#
            return actions, self.states, rewards, dones, goods

        rewards = np.vstack([
            fun_reward(s, None, g, False) for i, (s, g) in enumerate(zip(self.states, self.goals))
            ])
        return actions, self.states, rewards, dones, goods

    def goal_met(self, rewards):
        return rewards > 30.
        return rewards > .5
