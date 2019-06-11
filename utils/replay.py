import random, sys
import numpy as np

sys.path.append("PrioritizedExperienceReplay")
from PrioritizedExperienceReplay.proportional import Experience as Memory

from utils.learning_rate import LinearAutoSchedule as LinearSchedule

class ReplayBuffer:
    def __init__(self,
            n_step, replay_reanalyze,
            buffer_size, select_count, max_ep_draw_count,
            alpha, beta_base, beta_end, beta_horizon,
            recalc_delta
            ):
        self.batch_count = 0
        self.inds = None

        self.beta = LinearSchedule(beta_horizon, beta_base, beta_end)

        self.counter = 0
        self.ts = 0
        self.timetable = [0] * buffer_size

        self.n_step = n_step
        self.max_ep_draw_count = max_ep_draw_count
        self.delta = recalc_delta

        self.replay_reanalyze = replay_reanalyze
        self.buffer_size = buffer_size

        self.mem = Memory(buffer_size, select_count, alpha)

    def sample(self, batch_size, critic):
        self.inds, data = zip(*self._sample(batch_size, critic))
# lol TODO : kick off numpy vstack transpose
        data = np.vstack(data)
        return (data.T)

    def add(self, batch, prios, hashkey):
        if len(prios) < self.n_step * 2:
            return
        if not self._worth_experience(prios):
            return

        self.ts = (self.ts + 1) % len(self.timetable)
        # do first update when we do first freeze
        self.timetable[self.ts] = self.counter - self.delta
        for i, data in enumerate(batch):
            self.mem.add([np.asarray(data), i, len(prios) - i - 1, hashkey, self.ts], prios[i])

    def _worth_experience(self, prios):
#        return True
        if not len(self):
            return True
        if len(self) < self.buffer_size:
            return True
        for _ in range(10):
            data = self.mem.select(1.)
            if None == data:
                return True
            _, w, _ = data
            status = prios.mean() > np.mean(w)
            if status:
                return True
        return 0 == random.randint(0, 4)

    def _sample(self, batch_size, critic):
        """
        sampling should be multithreaded ~ mainly recalc
        """
        self.counter += 1
        self.batch_count = 0
        while self.batch_count < batch_size:
            data = self.mem.select(self.beta.value())
            if None == data:
                continue
            batch, _, inds = data
            if None == batch:
                continue
            _, local_forward, local_backward, hashkey, timestamp = zip(*batch)

            uniq = set(map(lambda i_b: i_b[0] - i_b[1], zip(inds, local_forward)))
            for i, b, f, k, t in zip(inds, local_backward, local_forward, hashkey, timestamp):
                pivot = i - f
                if pivot < 0 or pivot + b + f > len(self):
                    continue # temporarely we want to avoid this corner case .. TODO
                if pivot not in uniq:
                    continue
                if 0 != self.mem.tree.data[pivot][1]:
                    continue
                assert pivot + self.mem.tree.data[pivot][2] == i + b, "--> {} {} :: {} {}".format(pivot, [ (x[1], x[2]) for x in self.mem.tree.data[pivot:i] ], i, b)
                uniq.remove(pivot)
                bc = self.batch_count
                data = zip(*self._do_sample_wrap(pivot, b + f, critic, k, t))
                if bc == self.batch_count:
                    continue
                yield data

    def _do_sample_wrap(self, pivot, length, critic, hashkey, timestamp):
        return self._do_sample(self.mem.tree.data[pivot:pivot+length], pivot, length, critic, hashkey, timestamp)

# TODO :: REFACTOR! too complex logic, from replasy-reanal to recalc updates n timetable ..
    def _do_sample(self, full_episode, pivot, length, critic, hashkey, timestamp):
        available_range = range(length - self.n_step)

        top = min(len(available_range), self.max_ep_draw_count)
        replay = random.sample(available_range, random.randint(1, top))

        recalc = abs(self.timetable[timestamp] - self.counter) >= self.delta

        if not critic or not self.replay_reanalyze:
            episode_approved = map(lambda i: (full_episode[i][0], i in replay), range(length))
        else:
            episode_approved = critic.reanalyze_experience(full_episode, replay, recalc, -1 == timestamp)

        if recalc:
            self.timetable[timestamp] = self.counter

        for i, step_good in enumerate(episode_approved):
            step, good = step_good
            if recalc and -1 != timestamp and self.replay_reanalyze:
                self.mem.tree.data[pivot + i][0][...] = np.asarray(step)

            if i not in replay:
                continue
            if not good:
                continue
            self.batch_count += 1
            yield pivot + i, step

    def update(self, prios):
        '''
        replay buffer must be single thread style access, or properly locked ...
          ( sample, update, add )
          well in theory as it is not expanding, we dont care much of reads only .. for now lol ..
        '''
        self.mem.priority_update(np.hstack(self.inds), prios)
        self.inds = None

    def __len__(self):
        return len(self.mem)
