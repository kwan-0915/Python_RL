from environments.lunar_lander_wrapper import LunarLanderContinous
from utilities.env_wrapper import EnvWrapper


def create_env_wrapper(config):
    env_name = config['env']
    if env_name == "LunarLanderContinuous-v2":
        return LunarLanderContinous(config)

    return EnvWrapper(env_name)