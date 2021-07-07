from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common.utils import *

HIDDEN_SIZE = 256


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation = "tanh"):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim, HIDDEN_SIZE]
        sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, act_dim]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post, output_activation=output_activation)

    def forward(self, obs_batch):
        out = self.prev_dense(obs_batch)
        out = self.post_dense(out)
        return out


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim + act_dim, HIDDEN_SIZE]
        sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, 1]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post)

    def forward(self, obs_batch, action_batch):
        out = torch.cat((obs_batch, action_batch), dim=-1)
        out = self.prev_dense(out)
        out = self.post_dense(out)
        return out