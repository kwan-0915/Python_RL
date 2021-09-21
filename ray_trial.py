import ray
from collections import deque

@ray.remote
class SharedQueue:
    def __init__(self, length_1, length_2, length_3):
        self.queues = [deque(maxlen=length_1), deque(maxlen=length_2), deque(maxlen=length_3)]

    def get_queue(self, queue_id):
        if queue_id > len(self.queues): raise IndexError("Input id out of range")

        return self.queues[queue_id]

    def append(self, queue_id, num):
        if queue_id > len(self.queues): raise IndexError("Input id out of range")

        self.queues[queue_id].append(num)

@ray.remote
def fun1(queue_actor, queue_id, num):
    queue_actor.append.remote(queue_id=queue_id, num=num)

@ray.remote
def fun2(queue_actor, queue_id, num):
    queue_actor.append.remote(queue_id=queue_id, num=num*num)

if __name__ == '__main__':
    ray.init()

    share_queue_actor = SharedQueue.remote(30, 30, 30)

    print('Before append : ', ray.get(share_queue_actor.get_queue.remote(queue_id=0)))

    for i in range(10):
        fun1.remote(queue_actor=share_queue_actor, queue_id=0, num=i)
        fun2.remote(queue_actor=share_queue_actor, queue_id=0, num=i)

    while len(ray.get(share_queue_actor.get_queue.remote(queue_id=0))) < 10: pass

    print('After append : ', ray.get(share_queue_actor.get_queue.remote(queue_id=0)))

