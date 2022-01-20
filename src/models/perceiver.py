# adapted from https://github.com/deepmind/deepmind-research/blob/master/perceiver/perceiver.py
import math
from tracemalloc import is_tracing
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional_encoding import TrainablePositionEncoding


#  -----------------------------------------------------------
#  ----------------------  Primitives  -----------------------
#  -----------------------------------------------------------


# TODO: add attention mask
def attend(q, k, v, is_training, dropout_prob=0.0, attention_mask=None):
    """Computes multi-head attention using a query, key and value.

    Args:
        q: Query with shape [batch, q_indices, num_heads, head_dim].
        k: Key with shape [batch, kv_indices, num_heads, head_dim].
        v: Value with shape [batch, kv_indices, num_heads, head_dim].
            dropout_prob: dropout probability on the attention weights.
        dropout_prob: dropout probability on the attention weights.
        attention_mask: Array of shape [batch, q_indices, kv_indices] indicating
            which attentions are valid
    Returns:
        Output of the attention with shape [batch, q_indices, hiddens]
    """
    batch_size, q_indices, num_heads, q_head_dim = q.shape

    # 'bthd,bThd->bhtT'
    q = q.transpose(1, 2).contiguous()
    k = torch.permute(k, (0, 2, 3, 1)).contiguous()

    attention = torch.matmul(q, k)

    scale = 1. / math.sqrt(q_head_dim)
    attention *= scale

    # softmax on T dim, or index dim of value
    # [batch, heads, t, T]
    normalized = F.softmax(attention, dim=-1)

    if is_training:
        normalized = F.dropout(normalized, p=dropout_prob, training=is_training)

    # 'bhtT,bThv->bthv'
    v = v.transpose(1, 2).contiguous()
    summed = torch.matmul(normalized, v)
    summed = summed.transpose(1, 2).contiguous()
    summed = summed.view(batch_size, q_indices, -1)

    return summed


def conv_1d(in_features, out_features, bias=True, init_scale=1.0, dtype=None):
    """A 1D convolution."""
    conv_1d = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        dtype=dtype,
    )

    # init according to perceiver repo (variance scaling with truncated normal)
    stddev = np.sqrt(init_scale)

    # Adjust stddev for truncation.
    # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
    distribution_stddev = np.asarray(.87962566103423978, dtype=dtype)
    stddev = stddev / distribution_stddev

    nn.init.constant_(conv_1d.bias, 0)
    nn.init.trunc_normal_(conv_1d.weight, std=stddev)

    return conv_1d


def layer_norm(normalized_shape):
    """Layer norm."""
    return nn.LayerNorm(normalized_shape=normalized_shape)


#  -----------------------------------------------------------
#  -----------------------  Modules  -------------------------
#  -----------------------------------------------------------


class MLP(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(
        self,
        in_channels,
        widening_factor=4,
        dropout_prob=0.,
        init_scale=1.,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.widening_factor = widening_factor
        self.dropout_prob = dropout_prob
        self.init_scale = init_scale

        self.linear1 = conv_1d(
            in_features=in_channels,
            out_features=self.widening_factor * self.in_channels,
            init_scale=self.init_scale,
        )
        self.linear2 = conv_1d(
            in_features=self.widening_factor * self.in_channels,
            out_features=self.in_channels,
            init_scale=self.init_scale,
        )

    def forward(self, x, is_training):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = F.dropout(x, p=self.dropout_prob, training=is_training)

        return x


class Attention(nn.Module):
    """Multi-headed {cross, self}-attention."""

    def __init__(
        self,
        input_q_channels,
        input_kv_channels,
        num_heads=8,
        init_scale=1.,
        with_final_bias=True,
        final_init_scale_multiplier=1.,
        dropout_prob=0.,
        qk_channels=None,
        v_channels=None,
        output_channels=None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.init_scale = init_scale
        self.with_final_bias = with_final_bias
        self.final_init_scale = final_init_scale_multiplier * init_scale
        self.dropout_prob = dropout_prob

        # this is from the input
        self.input_q_channels = input_q_channels
        self.input_kv_channels = input_kv_channels

        # params for attention op
        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.output_channels = output_channels

        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if self.qk_channels is None:
            self.qk_channels = input_q_channels

        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if self.v_channels is None:
            self.v_channels = self.qk_channels

        # Project the output of QKV attention to a desired number of channels.
        # Default to the same number as the output of the QKV attention operation.
        if self.output_channels is None:
            self.output_channels = self.v_channels

        if self.qk_channels % self.num_heads != 0:
            raise ValueError(f'qk_channels ({self.qk_channels}) must be divisible by'
                f' num_heads ({self.num_heads}).')
        if self.v_channels % self.num_heads != 0:
            raise ValueError(f'v_channels ({self.v_channels}) must be divisible by'
                f' num_heads ({self.num_heads}).')

        # at this point these should be set correctly
        self.qk_channels_per_head = self.qk_channels // self.num_heads
        self.v_channels_per_head = self.v_channels // self.num_heads

        self.query_proj = conv_1d(
            in_features=input_q_channels,
            out_features=self.qk_channels,
            init_scale=self.init_scale,
        )
        self.key_proj = conv_1d(
            in_features=input_kv_channels,
            out_features=self.qk_channels,
            init_scale=self.init_scale,
        )
        self.value_proj = conv_1d(
            in_features=input_kv_channels,
            out_features=self.v_channels,
            init_scale=self.init_scale,
        )

        self.post_attn_proj = conv_1d(
            in_features=self.v_channels,
            out_features=self.output_channels,
            init_scale=self.final_init_scale,
            bias=self.with_final_bias,           
        )

    def forward(self, inputs_q, inputs_kv, is_training, attention_mask=None):
        q = self.query_proj(inputs_q)
        k = self.key_proj(inputs_kv)
        v = self.value_proj(inputs_kv)

        # multi-head attn
        batch_size, q_indices, _ = inputs_q.shape
        _, kv_indices, _ = inputs_kv.shape

        q = q.view(batch_size, q_indices, self.num_heads, self.qk_channels_per_head)
        k = k.view(batch_size, kv_indices, self.num_heads, self.qk_channels_per_head)
        v = v.view(batch_size, kv_indices, self.num_heads, self.v_channels_per_head)

        result = attend(
            q, k, v,
            is_training=is_training,
            dropout_prob=self.dropout_prob,
            attention_mask=attention_mask
        )

        return self.post_attn_proj(result)


class SelfAttention(nn.Module):
    """A self-attention module, including a dense block."""

    def __init__(
        self,
        input_channels,
        widening_factor=4,
        dropout_prob=0.,
        dropout_attn_prob=0.,
        num_heads=8,
        attn_init_scale=1.,
        dense_init_scale=1.,
        qk_channels=None,
        v_channels=None,
    ) -> None:
        super().__init__()

        self.widening_factor = widening_factor
        self.dropout_prob = dropout_prob
        self.dropout_attn_prob = dropout_attn_prob
        self.num_heads = num_heads
        self.attn_init_scale = attn_init_scale
        self.dense_init_scale = dense_init_scale

        self.input_channels = input_channels
        self.qk_channels = qk_channels
        self.v_channels = v_channels

        self.layer_norm1 = nn.LayerNorm(self.input_channels)
        self.layer_norm2 = nn.LayerNorm(self.input_channels)

        self.attn_layer = Attention(
            input_q_channels=self.input_channels,
            input_kv_channels=self.input_channels,
            num_heads=self.num_heads,
            init_scale=self.attn_init_scale,
            qk_channels=self.qk_channels,
            v_channels=self.v_channels,
        )

        self.mlp_layer = MLP(
            in_channels=self.attn_layer.output_channels,
            widening_factor=self.widening_factor,
            dropout_prob=self.dropout_prob,
            init_scale=self.dense_init_scale,
        )

    def forward(self, x, is_training, attention_mask=None):
        identity = x

        qkv_inputs = self.layer_norm1(x)
        attention = self.attn_layer(
            qkv_inputs, qkv_inputs,
            is_training=is_training, attention_mask=attention_mask
        )

        attention = F.dropout(attention, p=self.dropout_attn_prob, training=is_training)
        attention += identity

        identity = attention
        x = self.layer_norm2(attention)
        x = self.mlp_layer(x, is_training)
        x += identity

        return x


class CrossAttention(nn.Module):
    """A cross-attention module, including a dense block."""

    def __init__(
        self,
        input_q_channels,
        input_kv_channels,
        widening_factor=1,
        dropout_prob=0.,
        dropout_attn_prob=0.,
        num_heads=8,
        attn_init_scale=1.,
        dense_init_scale=1.,
        shape_for_attn='kv',
        use_query_residual=True,
        qk_channels=None,
        v_channels=None,
    ) -> None:
        super().__init__()

        if shape_for_attn not in ('kv', 'q'):
            raise ValueError('Invalid shape_for_attn!')

        self.widening_factor = widening_factor
        self.dropout_prob = dropout_prob
        self.dropout_attn_prob = dropout_attn_prob
        self.num_heads = num_heads
        self.attn_init_scale = attn_init_scale
        self.dense_init_scale = dense_init_scale
        self.shape_for_attn = shape_for_attn

        # this is from the input
        self.input_q_channels = input_q_channels
        self.input_kv_channels = input_kv_channels

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.use_query_residual = use_query_residual

        attn_output_channels = input_q_channels
        if self.shape_for_attn == 'kv':
            qk_channels = input_kv_channels
        elif self.shape_for_attn == 'q':
            qk_channels = input_q_channels

        v_channels = None
        if self.qk_channels is not None:
            qk_channels = self.qk_channels
        if self.v_channels is not None:
            v_channels = self.v_channels

        self.layer_norm1 = nn.LayerNorm(input_q_channels)  # x_query dim
        self.layer_norm2 = nn.LayerNorm(input_kv_channels)  # x_key_val_dim
        self.layer_norm3 = nn.LayerNorm(attn_output_channels)  # x_query dim

        # output_channels is same as query channels
        self.attn_layer = Attention(
            input_q_channels=input_q_channels,
            input_kv_channels=input_kv_channels,
            num_heads=self.num_heads,
            init_scale=self.attn_init_scale,
            dropout_prob=self.dropout_attn_prob,
            qk_channels=qk_channels,
            v_channels=v_channels,
            output_channels=attn_output_channels,
        )

        # in_channels here is same as query channels
        self.mlp_layer = MLP(
            in_channels=attn_output_channels,
            widening_factor=self.widening_factor,
            dropout_prob=self.dropout_prob,
            init_scale=self.dense_init_scale,
        )

    def forward(self, x_query, x_key_val, is_training, attention_mask=None):
        identity = x_query

        attention = self.attn_layer(
            self.layer_norm1(x_query),
            self.layer_norm2(x_key_val),
            is_training=is_training,
            attention_mask=attention_mask
        )

        attention = F.dropout(attention, p=self.dropout_attn_prob, training=is_training)

        # Optionally include a residual to the query.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self.use_query_residual:
            attention += identity

        identity = attention
        x = self.layer_norm3(attention)
        x = self.mlp_layer(x, is_training)
        x += identity

        return x


#  -----------------------------------------------------------
#  -----------------------  Perceiver  -----------------------
#  -----------------------------------------------------------


class PerceiverEncoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""

    def __init__(
        self,
        num_input_channels,
        # The encoder has a total of
        #   num_self_attends_per_block * num_blocks
        # self-attend layers. We share weights between blocks.
        num_self_attends_per_block=6,
        num_blocks=8,
        z_index_dim=512,
        num_z_channels=1024,
        qk_channels=None,
        v_channels=None,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        cross_attend_widening_factor=1,
        self_attend_widening_factor=1,
        dropout_prob=0.,
        z_pos_enc_init_scale=0.02,
        cross_attention_shape_for_attn='kv',
        use_query_residual=True,
    ) -> None:
        super().__init__()

        # z -> latent array params
        self.z_index_dim = z_index_dim
        self.num_z_channels = num_z_channels
        self.z_pos_enc_init_scale = z_pos_enc_init_scale

        self.num_blocks = num_blocks
        self.dropout_prob = dropout_prob
        self.qk_channels = qk_channels
        self.v_channels = v_channels

        # cross attn params
        self.num_input_channels = num_input_channels
        self.num_cross_attend_heads = num_cross_attend_heads
        self.cross_attend_widening_factor = cross_attend_widening_factor
        self.cross_attention_shape_for_attn = cross_attention_shape_for_attn
        self.use_query_residual = use_query_residual

        # self attn params
        self.num_self_attend_heads = num_self_attend_heads
        self.num_self_attends_per_block = num_self_attends_per_block
        self.self_attend_widening_factor = self_attend_widening_factor

        self.z_pos_enc = TrainablePositionEncoding(
            index_dims=self.z_index_dim,
            num_enc_channels=self.num_z_channels,
            init_scale=self.z_pos_enc_init_scale,
        )

        # perceiver io uses a single cross attend block
        self.cross_attend = CrossAttention(
            input_q_channels=self.num_z_channels,
            input_kv_channels=self.num_input_channels,  # num channels in input
            dropout_prob=self.dropout_prob,
            num_heads=self.num_cross_attend_heads,
            widening_factor=self.cross_attend_widening_factor,
            shape_for_attn=self.cross_attention_shape_for_attn,
            qk_channels=self.qk_channels,
            v_channels=self.v_channels,
            use_query_residual=self.use_query_residual,
        )

        self.self_attends = []
        for _ in range(self.num_self_attends_per_block):
            self.self_attends.append(
                SelfAttention(
                    input_channels=self.num_z_channels,
                    widening_factor=self.self_attend_widening_factor,
                    dropout_prob=self.dropout_prob,
                    num_heads=self.num_self_attend_heads,
                    qk_channels=self.qk_channels,
                    v_channels=self.v_channels,
                )
            )

    def latents(self, x):
        # return latent array repeated along batch dim
        return self.z_pos_enc(x)

    def forward(self, x, z, is_training, input_mask=None):
        # TODO: logic for input mask with cross attention
        z = self.cross_attend(z, x, is_training, input_mask)

        # perceiver io has removed interleaving cross attends
        for _ in range(self.num_blocks):
            for self_attend in self.self_attends:
                z = self_attend(z, is_training)

        return z


# TODO: enable weight sharing flag
class Perceiver(nn.Module):
    """The Perceiver: a scalable, fully attentional architecture."""

    def __init__(
        self,
        encoder,
        input_processor,
        output_processor,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.input_processor = input_processor
        self.output_processor = output_processor

    def forward(self, x, is_training, input_mask=None):
        if self.input_processor:
            # [b, index_dims, channels]
            x = self.input_processor(x)

        latent_array = self.encoder.latents(x)
        x = self.encoder(x, latent_array, is_training, input_mask)

        if self.output_processor:
            x = self.output_processor(x)

        return x


class PerceiverIO(nn.Module):
    """The PerceiverIO: a scalable, fully attentional architecture."""

    def __init__(
        self,
        encoder,
        decoder,
        input_processor,
        output_processor,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.input_processor = input_processor
        self.output_processor = output_processor

    def forward(self, x):
        return x
