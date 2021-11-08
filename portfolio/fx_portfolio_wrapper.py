import numpy as np
from signals.feat_controller import FeatController
from portfolio.portfolio_core import PortfolioEnv
from utilities.env_wrapper import ABCEnvWrapper

class FXPortfolioWrapper(ABCEnvWrapper):
    global_config = None

    def __init__(self, config):
        self.config = config
        FXPortfolioWrapper.global_config = config
        self.env_name = self.config['env']
        self.env = PortfolioEnv(self.config)
        self.action_space = self.env.action_space
        super().__init__(self.env_name, self.env)

    def reset(self):
        state = self.env.reset()
        return self.preprocess_state(state)

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_state, reward, done, info = self.env.step(self.normalise_action(action))
        return self.preprocess_state(next_state), reward, done, info

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        pass

    def plot(self, mode='human'):
        self.env.render(mode=mode)

    def close(self):
        self.env.close()

    def get_action_space(self):
        return self.env.action_space

    @staticmethod
    def preprocess_state(state):
        # 1) base settings
        seq_len = 24
        close_price = state[:, :, 3]
        local_config = FXPortfolioWrapper.global_config
        expected_indicators = local_config['indicators']
        feat_controller = FeatController(ohlcv_data=state, indicators=expected_indicators)

        # 2) feature settings
        features = feat_controller.get_feats()

        # ) add manual features
        # import datetime
        # dt = state[0, :, 5]
        # dt_start = datetime.datetime.utcfromtimestamp(float(dt[0]))
        # dt_end = datetime.datetime.utcfromtimestamp(float(dt[-1]))
        # print(f'run date: {dt_start} - {dt_end}')

        # ) get only last seq_len data in features
        for i in range(len(features)):
            features[i] = features[i][:, -seq_len:]
        #     print(f'feature {i} -- {expected_indicators[i]}:')
        #     print(features[i].shape)
        #     print(features[i])
        # raise ValueError

        # ) add random features if necessary
        num_rand_feat = 0
        for i in range(num_rand_feat):
            features.append(np.random.randn(close_price.shape[0], seq_len))

        # 3) combine features as state
        n_features = len(features)
        num_assets = close_price.shape[0]
        processed_state = np.concatenate(features, axis=0)
        processed_state = processed_state.reshape((n_features, num_assets, seq_len))
        processed_state = np.swapaxes(processed_state, 0, 1)
        processed_state = np.swapaxes(processed_state, 1, 2)
        # print(processed_state)
        # print(processed_state.shape)
        # raise ValueError

        processed_state = processed_state.flatten()
        return processed_state

    @staticmethod
    def normalise_action(action):
        return action / np.abs(action).sum()
        # return action

    def normalise_state(self, state):
        return state

    def normalise_reward(self, reward):
        return reward
