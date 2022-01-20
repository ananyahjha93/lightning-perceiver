import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
from typing import Any, Iterator, List, Dict, Tuple
from src.models.io_processors import (
    ImagePreprocessor,
    ClassificationPostprocessor,
)
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.models.perceiver import Perceiver, PerceiverEncoder
from src.models.positional_encoding import FourierPositionEncoding, TrainablePositionEncoding
from src.optimizers.lamb import LAMB
from src.optimizers.scheduler import no_warmup_cosine_decay
from src.datamodules.images.imagenet import ImagenetDataModule, imagenet_normalization


class ImageClassification(pl.LightningModule):
    def __init__(
        self,
        num_samples,
        batch_size,
        num_accelerators,
        learning_rate=0.002,
        weight_decay=0.,
        exclude_bn_bias=True,
        training_epochs=110,
        decay_start_from=55,
        num_classes=1000,
        img_size=224,
        num_blocks=8,
        z_index_dim=512,
        num_z_channels=1024,
        qk_channels=None,
        v_channels=None,
        num_self_attends_per_block=6,
        num_self_attend_heads=8,
        self_attend_widening_factor=1,
        cross_attend_widening_factor=1,
        num_cross_attend_heads=1,
        cross_attention_shape_for_attn='kv',
        use_query_residual=True,
        prep_type='pixels',
        spatial_downsample=1,
        conv1x1_num_img_channels=256,
        concat_or_add_pos='concat',
        position_encoding_type='fourier',
        trainable_enc_channels=256,
        trainable_enc_init_scale=0.02,
        fourier_enc_bands=64,
        fourier_enc_concat_pos=True,
        fourier_enc_sine_only=False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        # training params
        self.learning_rate = learning_rate
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.training_epochs = training_epochs
        self.decay_start_from = decay_start_from
        self.num_classes = num_classes

        global_batch_size = (
            num_accelerators * batch_size if num_accelerators > 0 else batch_size
        )
        self.train_iters_per_epoch = num_samples // global_batch_size

        # perceiver params
        self.num_blocks = num_blocks
        self.z_index_dim = z_index_dim
        self.num_z_channels = num_z_channels
        self.qk_channels = qk_channels
        self.v_channels = v_channels

        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_self_attend_heads = num_self_attend_heads
        self.self_attend_widening_factor = self_attend_widening_factor

        self.cross_attend_widening_factor = cross_attend_widening_factor
        self.num_cross_attend_heads = num_cross_attend_heads
        self.cross_attention_shape_for_attn = cross_attention_shape_for_attn
        self.use_query_residual = use_query_residual

        # preprocessor params
        self.prep_type = prep_type
        self.spatial_downsample = spatial_downsample
        self.conv1x1_num_img_channels = conv1x1_num_img_channels
        self.concat_or_add_pos = concat_or_add_pos

        # pos encoding params
        self.position_encoding_type = position_encoding_type
        self.index_dims = (img_size, img_size)
        self.trainable_enc_channels = trainable_enc_channels
        self.trainable_enc_init_scale = trainable_enc_init_scale

        self.fourier_enc_bands = fourier_enc_bands
        self.fourier_enc_concat_pos = fourier_enc_concat_pos
        self.fourier_enc_sine_only = fourier_enc_sine_only

        # accuracy metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy(compute_on_step=False)
        self.test_acc = torchmetrics.Accuracy(compute_on_step=False)

        # compute num_input_channels
        num_enc_channels = None

        position_encoding = None
        if self.position_encoding_type == 'trainable':
            num_enc_channels = self.trainable_enc_channels

            position_encoding = TrainablePositionEncoding(
                index_dims=self.index_dims,
                num_enc_channels=self.trainable_enc_channels,
                init_scale=self.trainable_enc_init_scale,
            )
        elif self.position_encoding_type == 'fourier':
            num_enc_channels = self.fourier_enc_bands * np.prod(self.index_dims)

            if not self.fourier_enc_sine_only:
                num_enc_channels *= 2

            if self.fourier_enc_concat_pos:
                num_enc_channels += 2

            position_encoding = FourierPositionEncoding(
                index_dims=self.index_dims,
                num_bands=self.fourier_enc_bands,
                concat_pos=self.fourier_enc_concat_pos,
                sine_only=self.fourier_enc_sine_only,
            )
        else:
            raise ValueError('Invalid position_encoding_type!')

        if self.prep_type == 'pixels':
            num_img_channels = 3
        elif self.prep_type == 'conv1x1':
            num_img_channels = self.conv1x1_num_img_channels

        num_input_channels = None
        if self.concat_or_add_pos == 'concat':
            num_input_channels = num_enc_channels + num_img_channels
        elif self.concat_or_add_pos == 'add':
            assert num_enc_channels == num_img_channels
            num_input_channels = num_enc_channels

        input_processor = ImagePreprocessor(
            position_encoding,
            prep_type=self.prep_type,
            spatial_downsample=self.spatial_downsample,
            num_img_channels=self.conv1x1_num_img_channels,
            concat_or_add_pos=self.concat_or_add_pos,
        )

        output_processor = ClassificationPostprocessor(
            representation_dim=self.num_z_channels,
            num_classes=self.num_classes,
        )

        perceiver_encoder = PerceiverEncoder(
            num_input_channels=num_input_channels,
            num_self_attends_per_block=self.num_self_attends_per_block,
            num_blocks=self.num_blocks,
            z_index_dim=self.z_index_dim,
            num_z_channels=self.num_z_channels,
            qk_channels=self.qk_channels,
            v_channels=self.v_channels,
            num_cross_attend_heads=self.num_cross_attend_heads,
            num_self_attend_heads=self.num_self_attend_heads,
            cross_attend_widening_factor=self.cross_attend_widening_factor,
            self_attend_widening_factor=self.self_attend_widening_factor,
            cross_attention_shape_for_attn=self.cross_attention_shape_for_attn,
            use_query_residual=self.use_query_residual,
        )

        self.model = Perceiver(
            perceiver_encoder,
            input_processor,
            output_processor,
        )

    def exclude_from_wt_decay_and_layer_adaptation(
        self,
        named_params: Iterator[Tuple[str, torch.Tensor]],
        weight_decay: float,
        skip_list: List[str] = ['bias', 'bn'],
    ) -> List[Dict]:
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{'params': params, 'weight_decay': weight_decay, 'exclude_from_layer_adaptation': False},
                {'params': excluded_params, 'weight_decay': 0., 'exclude_from_layer_adaptation': True}]

    def configure_optimizers(self):
        # TODO: what about learnable params?
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(
                self.named_parameters(), weight_decay=self.weight_decay
            )
        else:
            params = self.parameters()

        optimizer = LAMB(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        decay_after_steps = self.train_iters_per_epoch * self.decay_start_from
        total_steps = self.train_iters_per_epoch * self.training_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                no_warmup_cosine_decay(decay_after_steps, total_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


    def shared_step(self, batch):
        x, y = batch
        logits = self.model(x)

        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(F.softmax(logits, dim=1), y)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        self.log('train_acc_epoch', self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, y = self.shared_step(batch)
        self.val_acc(F.softmax(logits, dim=1), y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, y = self.shared_step(batch)
        self.test_acc(F.softmax(logits, dim=1), y)

        self.log('test_loss', loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

        return loss

if __name__ == '__main__':
    pl.seed_everything(1234)

    data_dir = ""
    ckpt_path = ""
    batch_size = 128
    num_workers = 8
    img_size = 224
    training_epochs = 110
    num_accelerators = 8

    # initialize datamodule
    dm = ImagenetDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=img_size,
    )

    num_samples = dm.num_samples
    normalization = imagenet_normalization()

    # initialize transforms
    # dm.train_transforms = TrainTransforms()
    # dm.val_transforms = EvalTransforms()

    # model init, checkpointing
    if ckpt_path == "":
        model = ImageClassification(
            num_samples=num_samples,
            batch_size=batch_size,
            num_accelerators=num_accelerators,
        )
    else:
        print('____________________________________________________')
        print('Loading model weights from checkpoint...')
        print('Args initialized according to this script')
        print('____________________________________________________')
        model = ImageClassification.load_from_checkpoint(ckpt_path, strict=False, **args.__dict__)

    # callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(save_last=True),
    ]

    # trainer
    trainer = pl.Trainer(
        max_epochs=training_epochs,
        accelerator="gpu",
        strategy="ddp",
        devices=num_accelerators,
        callbacks=callbacks,
        precision=32,
        ckpt_path=None if ckpt_path == "" else ckpt_path,
    )
    trainer.fit(model, dm)
