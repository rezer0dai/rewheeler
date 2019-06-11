# this we want to be as simple as possible, no additional logic nor assumptions
# just sharing what we saw .. replay buffs are forwarding us interestings stuffs
# sample therefore random
import random
from multiprocessing import Queue, Process
from utils.replay import *

from threading import Thread

class ExperienceObserver(Process):#Thread):#
    def __init__(self, size):
        super().__init__()

        self.exps = [None] * size
        self.keys = {}
        self.ind = 0

        self.cmd = { "add" : self._add, "sample" : self._sample }

        self.channel = Queue()
        self.sampler = Queue()

    def run(self):
        while True: # single thread is fine
            data = self.channel.get()
            cmd, data = data
            self.cmd[cmd](*data)

    def add(self, data, hashkey):
        self.channel.put(("add", (data, hashkey)))

    def sample(self):
        self.channel.put(("sample", ()))
        return self.sampler.get()

    def _add(self, data, key):
        if self._update(data, key):
            return
        self.keys[key] = self.ind
        self.exps[self.ind] = data
        self.ind = (self.ind + 1) % len(self.exps)

    def _update(self, data, key):
        if key not in self.keys:
            return False
        self.exps[ self.keys[key] ] = data
        return True

    def _sample(self):
        data = random.choice(self.exps[:len(self.keys)]) if len(self.keys) else None
        self.sampler.put(data)

class CrossExpBuffer(ReplayBuffer):
    def __init__(self, mgr,
            n_step, replay_reanalyze,
            buffer_size, select_count, max_ep_draw_count,
            alpha, beta_base, beta_end, beta_horizon,
            recalc_delta, share_exp_ratio):
        super().__init__(
            n_step, replay_reanalyze,
            buffer_size, 1 + int(select_count // (1 + share_exp_ratio)), max_ep_draw_count,
            alpha, beta_base, beta_end, beta_horizon,
            recalc_delta)
        self.mgr = mgr
        self.share_exp_ratio = share_exp_ratio

    def add(self, batch, prios, hashkey):
#        batch = [ [data, None, None, None, None] for data in batch ]
        batch = [ data for data in batch ]
        self.mgr.add([ [data, None, None, None, None] for data in batch ], hashkey)
        super().add(batch, prios, hashkey)
#        print("BATCHO SAMA :", np.array(batch).shape, np.array(np.array(batch)[0]).shape, len(self))

    def _do_sample(self, full_episode, pivot, length, critic, hashkey, timestamp):
        # forwarding already sampled data
        if not self.share_exp_ratio:
            for i, data in super()._do_sample(full_episode, pivot, length, critic, hashkey, timestamp):
                yield i, data

        self.mgr.add(full_episode, hashkey) # update latest record ( encoders )

        for _ in range(self.share_exp_ratio):
            # withdraw possibly cross-sampled data ( cross simulation / cross bot )
            data = self.mgr.sample()
            if None == data:
                return # nothin to be seen yet

            cross_episode = data
            for _, data in super()._do_sample(
                    cross_episode,
                    0, len(cross_episode),
                    critic, None, -1):
                yield -1, data # indicating we dont want to touch that data at update prios later on

    def update(self, prios):
        return # need to move to prio tree approach ( inherit and update )
        self.inds = np.hstack(self.inds).reshape(-1)
#        assert len(prios) == sum(map(lambda i: i >= 0, self.inds)), "FAIL" # good
        prios = prios[self.inds >= 0]
        self.inds = self.inds[self.inds >= 0]
        assert len(prios) == sum(map(lambda i: i >= 0, self.inds)), "FAIL" # nope
        super().update(prios)

def cross_exp_buffer(buffer_size):
    mgr = ExperienceObserver(buffer_size)
    def cross_buff(
            n_step, replay_reanalyze,
            buffer_size, select_count, max_ep_draw_count,
            alpha, beta_base, beta_end, beta_horizon,
            recalc_delta, share_exp_ratio):
        ceb = CrossExpBuffer(mgr,
            n_step, replay_reanalyze,
            buffer_size, select_count, max_ep_draw_count,
            alpha, beta_base, beta_end, beta_horizon,
            recalc_delta, share_exp_ratio)
        return ceb

    mgr.start() # separate process as we overhauling our main process with python threads
    return cross_buff
