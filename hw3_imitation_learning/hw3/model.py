"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


# TODO: Students implement ObstaclePolicy here. (DONE)
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 128,
        depth: int = 2,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model = d_model
        self.depth = depth
        flat_out = chunk_size * action_dim
        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, d_model))
            layers.append(nn.ReLU())
            in_dim = d_model
        layers.append(nn.Linear(in_dim, flat_out))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        flat = self.net(state)
        return flat.view(state.shape[0], self.chunk_size, self.action_dim)

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(state)

class ResidualBlock(nn.Module):
    """Simple residual MLP block."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 128,
        depth: int = 2,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.d_model = d_model
        self.depth = depth
        flat_out = chunk_size * action_dim
        self.structured_state_dim = 33
        self.use_structured = state_dim == self.structured_state_dim

        # Structured multicube path for the default multicube state layout:
        # [robot(6), red(7), green(7), blue(7), goal_one_hot(3), bin_pos(3)].
        if self.use_structured:
            cube_in_dim = 11  # cube(7) + relative position to bin(3) + distance(1)
            cube_emb_dim = max(32, d_model // 2)
            context_in_dim = 12  # robot(6) + goal(3) + bin_pos(3)

            self.cube_encoder = nn.Sequential(
                nn.Linear(cube_in_dim, cube_emb_dim),
                nn.ReLU(),
                nn.Linear(cube_emb_dim, cube_emb_dim),
                nn.ReLU(),
            )
            self.context_encoder = nn.Sequential(
                nn.Linear(context_in_dim, cube_emb_dim),
                nn.ReLU(),
                nn.Linear(cube_emb_dim, cube_emb_dim),
                nn.ReLU(),
            )

            # all cubes (3 * emb) + target cube emb + context emb
            fused_in_dim = 5 * cube_emb_dim
            self.fusion = nn.Sequential(
                nn.Linear(fused_in_dim, d_model),
                nn.ReLU(),
            )
            self.trunk = nn.Sequential(
                *[ResidualBlock(d_model) for _ in range(depth)]
            )
            self.head = nn.Linear(d_model, flat_out)
        else:
            # Fallback path for non-33D states to avoid shape assumptions.
            layers: list[nn.Module] = []
            in_dim = state_dim
            for _ in range(depth):
                layers.append(nn.Linear(in_dim, d_model))
                layers.append(nn.ReLU())
                in_dim = d_model
            layers.append(nn.Linear(in_dim, flat_out))
            self.fallback_net = nn.Sequential(*layers)


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if not self.use_structured:
            flat = self.fallback_net(state)
            return flat.view(state.shape[0], self.chunk_size, self.action_dim)

        robot_state = state[:, 0:6]
        goal_one_hot = state[:, 27:30]
        bin_pos = state[:, 30:33]

        # cubes: (B, 3, 7) in fixed order [red, green, blue]
        cubes = torch.stack(
            [state[:, 6:13], state[:, 13:20], state[:, 20:27]], dim=1
        )

        # Build cube features with relative geometry to the bin.
        cube_pos = cubes[:, :, :3]
        rel = cube_pos - bin_pos.unsqueeze(1)
        rel_dist = torch.norm(rel, dim=-1, keepdim=True)
        cube_feat = torch.cat([cubes, rel, rel_dist], dim=-1)  # (B, 3, 11)

        cube_emb = self.cube_encoder(cube_feat.view(-1, cube_feat.shape[-1]))
        cube_emb = cube_emb.view(state.shape[0], 3, -1)  # (B, 3, E)

        # Goal routing: weighted sum picks the target cube embedding.
        target_emb = torch.sum(cube_emb * goal_one_hot.unsqueeze(-1), dim=1)
        all_cubes_emb = cube_emb.reshape(state.shape[0], -1)
        context_emb = self.context_encoder(torch.cat([robot_state, goal_one_hot, bin_pos], dim=1))

        fused = torch.cat([all_cubes_emb, target_emb, context_emb], dim=1)
        x = self.fusion(fused)
        x = self.trunk(x)
        flat = self.head(x)
        return flat.view(state.shape[0], self.chunk_size, self.action_dim)

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int = 16,
    d_model: int = 128,
    depth: int = 2,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            # TODO: Build with your chosen specifications (DONE)
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            # TODO: Build with your chosen specifications (DONE)
        )
    raise ValueError(f"Unknown policy type: {policy_type}")