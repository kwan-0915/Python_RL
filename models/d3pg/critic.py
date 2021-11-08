import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """Critic - return Q value from given states and actions. """

    def __init__(self, n_features, seq_len, num_actions, hidden_size, init_w=3e-3, device='cuda'):
        """
        Args:
            num_states (int): state dimension
            num_actions (int): action dimension
            hidden_size (int): number of neurons in hidden layers
            init_w:
        """
        super(Critic, self).__init__()

        self.device = device
        self.n_features = n_features
        self.seq_len = seq_len
        self.num_actions = num_actions
        self.hidden_size = hidden_size

        conv_channel_size = 32
        k_size = 3
        n_layer = 2
        conv_output_size = seq_len
        for i in range(n_layer):
            conv_output_size = conv_output_size - k_size + 1
        conv_output_size = int(conv_output_size / k_size)  # max pool output size

        self.conv1d1 = nn.Conv1d(n_features, conv_channel_size, kernel_size=k_size)
        self.conv1d2 = nn.Conv1d(conv_channel_size, conv_channel_size, kernel_size=k_size)
        self.max_pooling1d1 = nn.MaxPool1d(kernel_size=k_size)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(num_actions * conv_channel_size * conv_output_size + num_actions, hidden_size)

        # self.linear1 = nn.Linear(num_states, hidden_size)
        # self.linear2 = nn.Linear(rnn_hidden_size * num_actions + num_actions, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.to(device)

    def forward(self, state, action):
        # x = torch.cat([state, action], dim=1)
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = self.linear3(x)

        # print(state.shape)
        # xs = F.relu(self.linear1(state))
        # print(xs.shape)
        # print(action.shape)
        # x = torch.cat((xs, action), dim=2)
        # print(x.shape)
        # x = F.relu(self.linear2(x))
        # x = self.linear3(x)

        # print('state: ')
        # print(state.shape)
        # print('action: ')
        # print(action.shape)
        # raise ValueError

        # 1) reshape the state (num_assets, seq_len, n_features) to LSTM input (batch, seq_len, n_features)
        # a) handling replay buffer batch
        if len(state.size()) == 1:
            rb_batch_size = 1
        elif len(state.size()) == 2:
            rb_batch_size = state.size()[0]
        else:
            raise ValueError('check input state shape: {}. sth wrong with state batch size.'.format(state.size()))

            # state = state.view((self.num_actions, self.seq_len, self.n_features))
        state = state.view((rb_batch_size * self.num_actions, self.seq_len, self.n_features))

        x = self.conv1d1(state.permute(0, 2, 1))
        x = self.conv1d2(x)
        x = self.max_pooling1d1(x)
        out = self.dropout(x)
        out = out.view(rb_batch_size, -1).squeeze()

        x = torch.cat([out, action], dim=-1)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x