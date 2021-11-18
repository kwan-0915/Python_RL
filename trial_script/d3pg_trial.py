import logging
import os
import sys
import ray
import copy
import yaml
import time
import argparse
import multiprocessing as mp
from datetime import datetime
from models.d3pg.agent import D3PGAgent
from models.d3pg.d3pg import D3PG
from models.d3pg.actor import Actor
from utilities.replay_buffer import D3PGReplayBuffer
from utilities.shared_actor import SharedActor
from utilities.logger import Logger

@ray.remote
def sampler_worker(config, shared_actor, log_dir=''):
    """
    Function that transfers replay to the buffer and batches from buffer to the queue.
    """
    batch_size = config['batch_size']

    # Logger
    logger = Logger(f"{log_dir}/data_struct")

    # Create replay buffer
    replay_buffer = D3PGReplayBuffer(int(config['replay_mem_size']))

    while ray.get(shared_actor.get_training_on.remote()):
        # (1) Transfer replays to global buffer
        n = len(ray.get(shared_actor.get_queue.remote("replay_queue")))

        for _ in range(n):
            replay = ray.get(shared_actor.get_queue.remote("replay_queue")).pop()
            replay_buffer.add(*replay)

        # (2) Transfer batch of replay from buffer to the batch_queue
        if len(replay_buffer) < batch_size:
            continue

        if config['replay_memory_prioritized']:
            try:
                inds, weights = ray.get(shared_actor.get_queue.remote("replay_priorities_queue")).pop()
                replay_buffer.update_priorities(inds, weights)
            except IndexError:
                sys.exit('Cannot load priority replay buffer')
        else:
            try:
                batch = replay_buffer.sample(batch_size)
                shared_actor.append.remote("batch_queue", batch)
            except KeyError:
                sys.exit('Batch Queue must not be None')

        # Log data structures sizes
        step = ray.get(shared_actor.get_update_step.remote())
        logger.scalar_summary("data_struct/global_episode", ray.get(shared_actor.get_global_episode.remote()), step)
        logger.scalar_summary("data_struct/replay_queue", len(ray.get(shared_actor.get_queue.remote("replay_queue"))), step)
        logger.scalar_summary("data_struct/batch_queue", len(ray.get(shared_actor.get_queue.remote("batch_queue"))), step)
        logger.scalar_summary("data_struct/replay_buffer", len(replay_buffer), step)

    if config['save_buffer_on_disk']: replay_buffer.dump(config["results_path"])

    print("Stop sampler worker.")
    shared_actor.set_child_threads.remote()

@ray.remote
def learner_worker(config, actor, target_actor, experiment_dir, shared_actor):
    learner = D3PG(config, actor, target_actor, shared_actor, log_dir=experiment_dir)
    learner.run()

@ray.remote
def agent_worker(config, policy, i, agent_type, experiment_dir, should_exploit=False, shared_actor=None):
    agent = D3PGAgent(config=config,
                      policy=policy,
                      n_agent=i,
                      agent_type=agent_type,
                      log_dir=experiment_dir,
                      should_exploit=should_exploit,
                      shared_actor=shared_actor)
    agent.run()

def timer(start, end):
    """ Helper to print training time """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def main(input_config=None):
    num_asset = input_config['num_asset'] + int(input_config['add_cash_asset'])  # get num of asset for first dim of state and action for replay buffer
    action_dim = num_asset * input_config["action_dim"]

    # Create directory for experiment
    experiment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/" + f"{config['results_path']}/{config['env']}-{config['model']}-{datetime.now():%Y-%m-%d_%H-%M-%S}"
    if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)

    # Shared object
    shared_actor = SharedActor.remote(replay_queue_size=input_config['replay_queue_size'], learner_w_queue_size=input_config['num_agents'], replay_priorities_queue_size=input_config['replay_priorities_queue_size'],
                                      batch_queue_size=input_config['batch_queue_size'], training_on=1, update_step=0, global_episode=0, n_threads=config['n_threads'])

    # Data sampler
    sampler_worker.remote(input_config, shared_actor, experiment_dir)

    # Learner (neural net training process)
    target_actor = Actor(input_config['n_features'], input_config["state_dim"], action_dim, input_config['dense_size'], device=input_config['device'])
    actor = copy.deepcopy(target_actor)
    actor_cpu = Actor(input_config['n_features'], input_config["state_dim"], action_dim, input_config['dense_size'], device=input_config['agent_device'])
    target_actor.share_memory()

    learner_worker.remote(input_config, actor, target_actor, experiment_dir, shared_actor)

    # Single agent for exploitation
    agent_worker.remote(input_config, target_actor, 0, "exploitation", experiment_dir, True, shared_actor)

    # Agents (exploration processes)
    for i in range(1, input_config['num_agents']):
        agent_worker.remote(input_config, actor_cpu, i, "exploration", experiment_dir, False, shared_actor)

    while not ray.get(shared_actor.get_child_threads.remote()): pass

    shared_actor.set_main_thread.remote()

    return shared_actor

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--config_file", type=str, help="Config file path")
    parser.add_argument("--num_cpus", type=int, help="Number of available CPUs")
    parser.add_argument("--num_gpus", type=int, help="Number of available GPUs")
    parser.add_argument("--data_file", type=str, help="Abs path for csv file")
    inputs = vars(parser.parse_args())

    t0 = time.time()

    with open(inputs['config_file'], "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
        config['path'] = inputs['data_file']
        ray.init(num_cpus=inputs['num_cpus'], num_gpus=inputs['num_gpus'])
        shared_actor = main(input_config=config)

    while not ray.get(shared_actor.get_main_thread.remote()): pass

    t1 = time.time()
    time.sleep(1.5)
    timer(t0, t1)

    print('End main thread')