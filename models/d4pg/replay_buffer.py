import os
import ray
import pickle
import random
import numpy as np
from collections import deque
from utilities.segment_tree import SumSegmentTree, MinSegmentTree

@ray.remote
class SharedObject:
    def __init__(self, replay_queue_size, learner_w_queue_size, replay_priorities_queue_size, batch_queue_size,
                 training_on, update_step, global_episode, n_threads):
        self.queues = {
            "replay_queue": deque(maxlen=replay_queue_size),
            "learner_w_queue": deque(maxlen=learner_w_queue_size),
            "replay_priorities_queue": deque(maxlen=replay_priorities_queue_size),
            "batch_queue": deque(maxlen=batch_queue_size)
        }

        self.training_on = training_on
        self.update_step = update_step
        self.global_episode = global_episode

        self.child_threads = [False] * n_threads
        self.ptr = 0

        self.main_thread = False

    def get_main_thread(self):
        return self.main_thread

    def set_main_thread(self):
        self.main_thread = True

    def get_child_threads(self):
        return all(self.child_threads)

    def set_child_threads(self):
        try:
            self.child_threads[self.ptr] = True
            self.ptr += 1
        except IndexError:
            pass

    def get_training_on(self):
        return self.training_on

    def set_training_on(self, num):
        self.training_on = num

    def get_update_step(self):
        return self.update_step

    def set_update_step(self):
        self.update_step += 1

    def get_global_episode(self):
        return self.global_episode

    def get_queue(self, queue_key):
        if queue_key not in self.queues.keys(): raise KeyError("No such key")

        return self.queues[queue_key]

    def append(self, queue_key, data):
        if queue_key not in self.queues.keys(): raise KeyError("No such key")

        self.queues[queue_key].append(data)

class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create replay buffer.
        Args:
            size (int): max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, gamma):
        data = (obs_t, action, reward, obs_tp1, done, gamma)

        self._storage.append(data)

        self._next_idx += 1

    def remove(self, num_samples):
        del self._storage[:num_samples]
        self._next_idx = len(self._storage)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, gammas = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, gamma = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            gammas.append(gamma)
        return [np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(
            gammas)]

    def sample(self, batch_size, **kwags):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        gammas: np.array
            product of gammas for N-step returns
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        weights = np.zeros(len(idxes))
        inds = np.zeros(len(idxes))
        return self._encode_sample(idxes) + [weights, inds]

    def dump(self, save_dir):
        fn = os.path.join(save_dir, "replay_buffer.pkl")
        with open(fn, 'wb') as f:
            pickle.dump(self._storage, f)
        print(f"Buffer dumped to {fn}")


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        self.it_capacity = 1
        while self.it_capacity < size * 2:  # We use double the soft capacity of the PER for the segment trees to allow for any overflow over the soft capacity limit before samples are removed
            self.it_capacity *= 2

        self._it_sum = SumSegmentTree(self.it_capacity)
        self._it_min = MinSegmentTree(self.it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        assert idx < self.it_capacity, "Number of samples in replay memory exceeds capacity of segment trees. Please increase capacity of segment trees or increase the frequency at which samples are removed from the replay memory"

        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def remove(self, num_samples):
        super().remove(num_samples)
        self._it_sum.remove_items(num_samples)
        self._it_min.remove_items(num_samples)

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        gammas: np.array
            product of gammas for N-step returns
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def dump(self, save_dir):
        fn = os.path.join(save_dir, "replay_buffer.pkl")
        with open(fn, 'wb') as f:
            pickle.dump(self._storage, f)
        print(f"Buffer dumped to {fn}")


def create_replay_buffer(config):
    size = config['replay_mem_size']
    if config['replay_memory_prioritized']:
        alpha = config['priority_alpha']
        return PrioritizedReplayBuffer(size=size, alpha=alpha)
    return ReplayBuffer(size)