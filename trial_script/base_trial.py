import gym
import yaml
import torch
import argparse
import numpy as np
from models.td3.td3 import TD3
from models.ddpg.ddpg import DDPG
from itertools import count

def main(config, model):
    args = argparse.Namespace(**config)
    env = gym.make(args.env_name)

    if args.seed:
        env.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    directory = "./exp/" + args.env_name + "./"

    agent = None
    if model == 'ddpg': agent = DDPG(state_dim, action_dim, max_action, args, directory)
    elif model == 'td3': agent = TD3(state_dim, action_dim, max_action, args, directory)

    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >= args.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':
        if args.load: agent.load()
        total_step = 0
        for i in range(args.max_episode):
            total_reward = 0
            step = 0
            state = env.reset()
            for _ in count():
                action = agent.select_action(state)
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
                next_state, reward, done, info = env.step(action)

                if args.render and i >= args.render_interval: env.render()
                agent.replay_buffer.add((state, next_state, action, reward, np.float64(done)))
                state = next_state
                if done: break

                step += 1
                total_reward += reward

            total_step += step + 1
            print("Total T:{} \tEpisode: {} \tTotal Reward: {:0.2f}".format(total_step, i, total_reward))
            agent.update()

            if i % args.log_interval == 0:
                agent.save()
    else:
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--model", type=str, help="Model to be ran")
    parser.add_argument("--config_file", type=str, help="Config file path")
    inputs = vars(parser.parse_args())

    with open(inputs['config_file'], "r") as file:
        main(config=yaml.load(file, Loader=yaml.SafeLoader), model=inputs['model'])
