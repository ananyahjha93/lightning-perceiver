# adapted from https://github.com/deepmind/deepmind-research/blob/master/perceiver/position_encoding.py

import math
import torch
import torch.nn as nn
import numpy as np


class TrainablePositionEncoding(nn.Module):
    """Trainable position encoding."""

    def __init__(
        self,
        index_dims,
        num_enc_channels=128,
        init_scale=0.02,
        **kwargs,
    ) -> None:
        super().__init__()

        self.index_dims = index_dims
        self.num_enc_channels = num_enc_channels
        self.init_scale = init_scale

        if type(index_dims) != int and len(index_dims) > 1:
            index_dims = np.prod(index_dims)

        self.encodings = torch.zeros([index_dims, num_enc_channels])
        self.encodings = nn.Parameter(self.encodings)

        # initialize
        nn.init.trunc_normal_(self.encodings, std=self.init_scale)

    def forward(self, x):
        batch_size = x.shape[0]
        encodings = self.encodings.unsqueeze(0).repeat(batch_size, 1, 1)
        encodings = encodings.to(x.get_device())

        return encodings


class FourierPositionEncoding(nn.Module):
    """Fourier (Sinusoidal) position encoding.

    Args:
        index_dims:
        num_bands:
        output_range:
        concat_pos:
        sine_only:
    """

    def __init__(
        self,
        index_dims,
        num_bands,
        output_range=(-1.0, 1.0),
        concat_pos=True,
        sine_only=False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.index_dims = index_dims
        self.num_bands = num_bands
        self.output_range = output_range
        self.concat_pos = concat_pos
        self.sine_only = sine_only

        # generate [-1, 1] coordinates
        # self.pos is tensor containing tensors of
        # shape (index_dims[0], ..., index_dims[-1], N), where N is len(index_dims)
        pos = self._build_linear_positions()

        # change shape to [index_dims[0] * ... * index_dims[-1], N]
        pos = pos.view(-1, pos.shape[-1])

        # enc is [index_dims[0] * ... * index_dims[-1], N * num_bands * 2 + len(index_dims)]
        self.encodings = self._generate_fourier_features(pos)

    def _build_linear_positions(self):
        """Generate an array of position indices for an N-D input array."""
        dim_ranges = [
            torch.linspace(self.output_range[0], self.output_range[1], steps=n_xels_per_dim)
            for n_xels_per_dim in self.index_dims
        ]

        array_index_grid = torch.meshgrid(*dim_ranges)
        return torch.stack(array_index_grid, dim=len(self.index_dims))

    def _generate_fourier_features(self, pos):
        """Generate a Fourier frequency position encoding with linear spacing."""
        min_freq = 1.0
        # Nyquist frequency at the target resolution:

        freq_bands = torch.stack(
            [
                torch.linspace(min_freq, res / 2, steps=self.num_bands)
                for res in self.index_dims
            ],
            dim=0
        )

        # pos [index_dims[0] * ... * index_dims[-1], N]
        # freq_bands [N, num_bands]
        # per_pos_features [index_dims[0] * ... * index_dims[-1], N, num_bands]
        per_pos_features = pos[:, :, None] * freq_bands[None, :, :]

        # per_pos_features [index_dims[0] * ... * index_dims[-1], N * num_bands]
        per_pos_features = per_pos_features.view(per_pos_features.shape[0], -1)

        if self.sine_only:
            per_pos_features = torch.sin(math.pi * per_pos_features)
        else:
            per_pos_features = torch.cat(
                [
                    torch.sin(math.pi * per_pos_features),
                    torch.cos(math.pi * per_pos_features)
                ],
                dim=-1
            )

        if self.concat_pos:
            per_pos_features = torch.cat([pos, per_pos_features], dim=-1)

        return per_pos_features

    def forward(self, x):
        batch_size = x.shape[0]
        encodings = self.encodings.unsqueeze(0).repeat(batch_size, 1, 1)
        encodings = encodings.to(x.get_device())

        return encodings


# TODO:
class PositionEncodingProjector(nn.Module):
    """Projects a position encoding to a target size."""

    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.encoding_projector = nn.Linear(input_size, output_size)

    def forward(self, base_position_encoding):
        return self.encoding_projector(base_position_encoding)
