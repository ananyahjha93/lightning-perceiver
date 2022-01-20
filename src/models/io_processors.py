import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional_encoding import TrainablePositionEncoding, FourierPositionEncoding
from typing import Optional


class ImagePreprocessor(nn.Module):
    def __init__(
        self,
        position_encoding,
        prep_type='pixels',
        spatial_downsample: int = 1,
        num_img_channels: int = 64,
        concat_or_add_pos: str = 'concat',
    ) -> None:
        super().__init__()

        self.position_encoding = position_encoding

        self.prep_type = prep_type
        self.spatial_downsample = spatial_downsample
        self.num_img_channels = num_img_channels
        self.concat_or_add_pos = concat_or_add_pos

        if prep_type not in ('conv', 'patches', 'pixels', 'conv1x1'):
            raise ValueError('Invalid prep_type!')

        if concat_or_add_pos not in ['concat', 'add']:
            raise ValueError(
                f'Invalid value {concat_or_add_pos} for concat_or_add_pos.')

        if self.prep_type == 'pixels':
            self.conv = None
        elif self.prep_type == 'conv1x1':
            self.conv = nn.Conv2d(
                in_channels=3,
                out_channels=self.num_img_channels,
                kernel_size=1,
                stride=self.spatial_downsample,
            )
        else:
            raise NotImplementedError('prep_type not implemented yet!')

    def _build_network_inputs(self, x):
        batch_size = x.shape[0]
        num_channels = x.shape[1]
        index_dims = np.prod(x.shape[2:])

        assert index_dims == np.prod(self.position_encoding.index_dims)

        # x to [b, prod(index_dims), c]
        x = x.view(batch_size, num_channels, index_dims)
        x = torch.permute(x, (0, 2, 1))

        pos = self.position_encoding(x)

        if self.concat_or_add_pos == 'concat':
            x = torch.cat([x, pos], dim=-1)
        elif self.concat_or_add_pos == 'add':
            # num img channels should be equal to pos channels
            # cannot use add with pixels, need conv1x1
            assert pos.shape[-1] == x.shape[-1]
            x += pos

        return x

    def forward(self, x):
        # x is an image [b, c, h, w]
        if self.prep_type == 'pixels':
            # crude downsample if downsample required in pixel mode
            # avoid this option and use transforms to downsample
            x = x[:, :, ::self.spatial_downsample, ::self.spatial_downsample]
        elif self.prep_type == 'conv1x1':
            x = self.conv(x)

        return self._build_network_inputs(x)


class ClassificationPostprocessor(nn.Module):
    def __init__(
        self,
        representation_dim,
        num_classes,
    ) -> None:
        super().__init__()

        self.classification_layer = nn.Linear(representation_dim, num_classes)

    def forward(self, x):
        # [b, index_dims, num_channels]
        return self.classification_layer(x[:, 0, :])
