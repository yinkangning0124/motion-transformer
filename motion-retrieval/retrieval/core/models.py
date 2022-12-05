from typing import List
import torch

from retrieval.utils.pytorch_util import PyTorchModule
from retrieval.core.nets import Mlp


class MuZeroFullyConnected(PyTorchModule):
    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 64,
        condition_on_target: bool = False,
        history_stack_size: int = 1,
        hidden_sizes: List[int] = [256, 256],
    ):
        self.save_init_params(locals())
        super().__init__()

        self.condition_on_target = condition_on_target
        self.embed_network = Mlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim * history_stack_size if not condition_on_target else obs_dim * (history_stack_size + 1),
            output_size=embed_dim,
        )
        self.dynamics_network = Mlp(
            hidden_sizes=hidden_sizes,
            input_size=embed_dim,
            output_size=embed_dim,
        )
        self.obs_pred_network = Mlp(
            hidden_sizes=hidden_sizes,
            input_size=embed_dim,
            output_size=obs_dim,
        )

    def prediction(self, encoded_state: torch.Tensor):
        pred_obs = self.obs_pred_network(encoded_state)
        return pred_obs

    def representation(self, observation: torch.Tensor):
        encoded_state = self.embed_network(observation)
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state: torch.Tensor):
        next_encoded_state = self.dynamics_network(encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized

    def initial_inference(self, observation: torch.Tensor):
        encoded_state = self.representation(observation)
        next_obs = self.prediction(encoded_state)
        return next_obs, encoded_state

    def recurrent_inference(self, encoded_state: torch.Tensor):
        next_encoded_state = self.dynamics(encoded_state)
        next_obs = self.prediction(next_encoded_state)
        return next_obs, next_encoded_state
