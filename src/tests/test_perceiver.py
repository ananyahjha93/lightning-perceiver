from operator import index
from tracemalloc import is_tracing
import torch

from src.models.io_processors import ImagePreprocessor, ClassificationPostprocessor
from src.models.positional_encoding import TrainablePositionEncoding, FourierPositionEncoding
from src.models.perceiver import Perceiver, PerceiverEncoder


def generate_image_batch():
    batch_size = 7
    num_channels = 3
    img_size = 224
    num_classes = 1000

    x = torch.zeros([batch_size, num_channels, img_size, img_size])
    y = torch.zeros([batch_size, num_classes])

    return x, y


def generate_image_preprocessor(
    pos_enc, prep_type, concat_or_add_pos, num_img_channels=None
):
    return ImagePreprocessor(
        position_encoding=pos_enc,
        prep_type=prep_type,
        concat_or_add_pos=concat_or_add_pos,
        num_img_channels=num_img_channels,
    )

def generate_perceiver_model(
    num_input_channels,
    pos_enc,
    prep_type,
    concat_or_add_pos,
    num_img_channels=None,
):
    num_self_attends_per_block = 6
    num_blocks = 8
    z_index_dim = 512
    num_z_channels = 1024

    num_cross_attend_heads = 1
    num_self_attend_heads = 8
    cross_attend_widening_factor = 1
    self_attend_widening_factor = 1

    cross_attention_shape_for_attn = 'kv'
    use_query_residual = True

    preprocessor = generate_image_preprocessor(
        pos_enc, prep_type, concat_or_add_pos, num_img_channels=num_img_channels
    )

    postprocessor = ClassificationPostprocessor(
        representation_dim=num_z_channels,
        num_classes=1000,
    )

    encoder = PerceiverEncoder(
        num_input_channels=num_input_channels,
        num_self_attends_per_block=num_self_attends_per_block,
        num_blocks=num_blocks,
        z_index_dim=z_index_dim,
        num_z_channels=num_z_channels,
        qk_channels=None,
        v_channels=None,
        num_cross_attend_heads=num_cross_attend_heads,
        num_self_attend_heads=num_self_attend_heads,
        cross_attend_widening_factor=cross_attend_widening_factor,
        self_attend_widening_factor=self_attend_widening_factor,
        cross_attention_shape_for_attn=cross_attention_shape_for_attn,
        use_query_residual=use_query_residual,
    )

    perceiver = Perceiver(
        encoder=encoder,
        input_processor=preprocessor,
        output_processor=postprocessor,
    )

    return perceiver


def test_perceiver_fourier_enc_with_pixels():
    # main run
    x, y = generate_image_batch()

    index_dims = (224, 224)
    num_bands = 64
    sine_only = False
    num_img_channels = 3
    concat_pos = True

    num_enc_channels = len(index_dims) * num_bands
    if not sine_only:
        num_enc_channels *= 2
    if concat_pos:
        num_enc_channels += 2

    pos_enc = FourierPositionEncoding(
        index_dims=index_dims,
        num_bands=64,
        concat_pos=concat_pos,
        sine_only=sine_only,
    )

    perceiver = generate_perceiver_model(
        num_input_channels=num_enc_channels + num_img_channels,
        pos_enc=pos_enc,
        prep_type='pixels',
        concat_or_add_pos='concat'
    )

    output = perceiver(x, is_training=True)
    # [batch, num_classes]
    assert output.shape == (7, 1000)


def test_perceiver_learned_enc_with_conv1x1():
    x, y = generate_image_batch()

    num_enc_channels = 256
    num_img_channels = 256

    pos_enc = TrainablePositionEncoding(
        index_dims=(224, 224),
        num_enc_channels=num_enc_channels,
        init_scale=0.02,
    )

    perceiver = generate_perceiver_model(
        num_input_channels=num_enc_channels + num_img_channels,
        pos_enc=pos_enc,
        prep_type='conv1x1',
        concat_or_add_pos='concat',
        num_img_channels=num_img_channels,
    )

    output = perceiver(x, is_training=True)
    # [batch, num_classes]
    assert output.shape == (7, 1000)


if __name__ == '__main__':
    test_perceiver_fourier_enc_with_pixels()
    test_perceiver_learned_enc_with_conv1x1()
