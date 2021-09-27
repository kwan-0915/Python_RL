import ray
from collections import deque

@ray.remote
class SharedActor:
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