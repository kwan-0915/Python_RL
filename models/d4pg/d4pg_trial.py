import argparse
import copy
import os
import ray
import yaml
import time
import multiprocessing as mp
from datetime import datetime
from time import sleep
from agent import Agent
from d4pg import D4PG
from actor import Actor
from replay_buffer import create_replay_buffer, SharedObject
from utilities.logger import Logger

@ray.remote
def sampler_worker(config, shared_object_actor, log_dir=''):
    """
    Function that transfers replay to the buffer and batches from buffer to the queue.
    """
    batch_size = config['batch_size']

    # Logger
    logger = Logger(f"{log_dir}/data_struct")

    # Create replay buffer
    replay_buffer = create_replay_buffer(config)

    while ray.get(shared_object_actor.get_training_on.remote()):
        # (1) Transfer replays to global buffer
        n = len(ray.get(shared_object_actor.get_queue.remote("replay_queue")))

        for _ in range(n):
            replay = ray.get(shared_object_actor.get_queue.remote("replay_queue")).pop()
            replay_buffer.add(*replay)

        # (2) Transfer batch of replay from buffer to the batch_queue
        if len(replay_buffer) < batch_size:
            continue

        try:
            inds, weights = ray.get(shared_object_actor.get_queue.remote("replay_priorities_queue")).pop()
            replay_buffer.update_priorities(inds, weights)
        except IndexError:
            pass

        try:
            batch = replay_buffer.sample(batch_size)
            shared_object_actor.append.remote("batch_queue", batch)
        except:
            sleep(0.1)
            continue

    if config['save_buffer_on_disk']:
        replay_buffer.dump(config["results_path"])

    print("Stop sampler worker.")

@ray.remote
def learner_worker(config, policy, target_policy_net, experiment_dir, shared_object_actor):
    learner = D4PG(config, policy, target_policy_net, shared_object_actor, log_dir=experiment_dir)
    learner.run()

@ray.remote
def agent_worker(config, policy, i, agent_type, experiment_dir, should_exploit=False, shared_object_actor=None):
    agent = Agent(config=config,
                  policy=policy,
                  n_agent=i,
                  agent_type=agent_type,
                  log_dir=experiment_dir,
                  should_exploit=should_exploit,
                  shared_object_actor=shared_object_actor)
    agent.run()

def main(input_config=None):
    batch_queue_size = input_config['batch_queue_size']
    n_agents = input_config['num_agents']

    # Create directory for experiment
    experiment_dir = f"{input_config['results_path']}/{input_config['env']}-{input_config['model']}-{datetime.now():%Y-%m-%d_%H-%M-%S}"
    if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

    # Shared object
    shared_object_actor = SharedObject.remote(replay_queue_size=64, learner_w_queue_size=n_agents, replay_priorities_queue_size=64,
                                              batch_queue_size=batch_queue_size, training_on=1, update_step=0, global_episode=0)

    # Data sampler
    sampler_worker.remote(input_config, shared_object_actor, experiment_dir)

    # Learner (neural net training process)
    target_policy_net = Actor(input_config['state_dim'], input_config['action_dim'], input_config['dense_size'], device=input_config['device'])
    policy_net = copy.deepcopy(target_policy_net)
    policy_net_cpu = Actor(input_config['state_dim'], input_config['action_dim'], input_config['dense_size'], device=input_config['agent_device'])
    target_policy_net.share_memory()

    learner_worker.remote(input_config, policy_net, target_policy_net, experiment_dir, shared_object_actor)

    # Single agent for exploitation
    agent_worker.remote(input_config, target_policy_net, 0, "exploitation", experiment_dir, True, shared_object_actor)

    # Agents (exploration processes)
    for i in range(1, n_agents):
        agent_worker.remote(input_config, policy_net_cpu, i, "exploration", experiment_dir, False, shared_object_actor)

    while not ray.get(shared_object_actor.get_should_exit.remote()): pass

    print("End.")

def timer(start, end):
    """ Helper to print training time """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--config_file", type=str, help="Config file path")
    parser.add_argument("--num_cpus", type=int, help="Number of available CPUs")
    parser.add_argument("--num_gpus", type=int, help="Number of available GPUs")
    inputs = vars(parser.parse_args())

    t0 = time.time()

    with open(inputs['config_file'], "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
        ray.init(num_cpus=inputs['num_cpus'], num_gpus=inputs['num_gpus'])
        main(input_config=config)

    t1 = time.time()
    time.sleep(1.5)
    timer(t0, t1)

