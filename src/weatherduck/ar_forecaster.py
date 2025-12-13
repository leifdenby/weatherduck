from typing import Optional

import torch
from torch import nn
from torch_geometric.data import HeteroData

from .step_predictor import EncodeProcessDecodeModel

__all__ = ["AutoRegressiveForecaster"]


class AutoRegressiveForecaster(nn.Module):
    """
    Auto-regressive forecaster that rolls out multiple steps using a
    step-wise predictor (EncodeProcessDecodeModel).
    """

    def __init__(self, step_predictor: EncodeProcessDecodeModel):
        super().__init__()
        self.step_predictor = step_predictor

    def forward(self, graph: HeteroData) -> torch.Tensor:
        """
        Perform an autoregressive rollout using the step_predictor.

        Parameters
        ----------
        graph : HeteroData
            Must contain on 'data' nodes:
              - x_init_states: [N, d_state, 2] initial history
              - x_target_states: [N, d_state, T] targets per step
              - x_forcing: [N, d_forcing, T]
              - x_static: [N, d_static]
            Edge structure must satisfy the step_predictor requirements.

        Returns
        -------
        torch.Tensor
            Predicted states of shape [T, N, d_state_out].
        """
        x_init = graph["data"].x_init_states  # [N, d_state, 2]
        x_targets = graph["data"].x_target_states  # [N, d_state, T]
        x_forcing = graph["data"].x_forcing  # [N, d_forcing, T]
        x_static = graph["data"].x_static  # [N, d_static]

        N, d_state, T = x_targets.shape
        d_forcing = x_forcing.shape[1]
        d_static = x_static.shape[1]

        assert x_init.shape == (N, d_state, 2)
        assert x_forcing.shape == (N, d_forcing, T)
        assert x_static.shape == (N, d_static)

        preds = []
        prev_states = x_init

        for t in range(T):
            current_state = prev_states[:, :, -1]  # [N, d_state]
            data_feats = torch.cat([current_state, x_forcing[:, :, t], x_static], dim=-1)

            step_graph = graph.clone()
            step_graph["data"].x = data_feats

            pred = self.step_predictor(step_graph)  # [N, d_state_out]
            preds.append(pred)

            prev_states = torch.cat([prev_states[:, :, 1:], pred.unsqueeze(-1)], dim=2)

        return torch.stack(preds, dim=2)
