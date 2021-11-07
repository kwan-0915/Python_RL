from environments.lunar_lander_wrapper import LunarLanderContinous
from utilities.env_wrapper import EnvWrapper
from glob import glob
import os
import imageio


def create_env_wrapper(config):
    env_name = config['env']
    if env_name == "LunarLanderContinuous-v2":
        return LunarLanderContinous(config)

    return EnvWrapper(env_name)

def make_gif(source_dir, output):
    """
    Make gif file from set of .jpeg images.
    Args:
        source_dir (str): path with .jpeg images
        output (str): path to the output .gif file
    Returns: None
    """
    batch_sort = lambda s: int(s[s.rfind('/')+1:s.rfind('.')])
    image_paths = sorted(glob(os.path.join(source_dir, "*.png")), key=batch_sort)

    images = []
    for filename in image_paths:
        images.append(imageio.imread(filename))

    imageio.mimsave(output, images)