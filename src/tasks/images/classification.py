# mixup adapted from https://github.com/facebookresearch/deit/blob/main/main.py
from parso import parse
import torch
import torchmetrics
import argparse
import torch.nn.functional as F
import pytorch_lightning as pl

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from typing import Any, Iterator, List, Dict, Tuple
from src.models.io_processors import (
    ImagePreprocessor,
    ClassificationPostprocessor,
)
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.models.perceiver import Perceiver, PerceiverEncoder
from src.models.positional_encoding import FourierPositionEncoding, TrainablePositionEncoding
from src.optimizers.lamb import LAMB
from src.optimizers.scheduler import linear_warmup_decay, no_warmup_cosine_decay
from src.datamodules.images.imagenet import ImagenetDataModule, imagenet_normalization
from src.transforms.images.imagenet_classification import EvalTransform, TrainTransform


class ImageClassification(pl.LightningModule):
    def __init__(
        self,
        num_samples,
        batch_size,
        num_accelerators,
        smoothing=0.1,
        optimizer='lamb',
        scheduler='no_warmup',
        learning_rate=0.002,
        weight_decay=0.1,
        exclude_bn_bias=True,
        training_epochs=110,
        decay_start_from=55,
        warmup_epochs=5,
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
        mixup=0.8,
        cutmix=1.,
        cutmix_minmax=None,
        mixup_prob=1.0,
        mixup_switch_prob=0.5,
        mixup_mode='batch',
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        # training params
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.learning_rate = learning_rate
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.training_epochs = training_epochs
        self.decay_start_from = decay_start_from
        self.warmup_epochs = warmup_epochs
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

        # mixup args
        self.mixup = mixup
        self.cutmix = cutmix
        self.cutmix_minmax = cutmix_minmax
        self.mixup_prob = mixup_prob
        self.mixup_switch_prob = mixup_switch_prob
        self.mixup_mode = mixup_mode
        self.smoothing = smoothing

        mixup_active = mixup > 0 or cutmix > 0. or cutmix_minmax is not None
        if mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=self.mixup,
                cutmix_alpha=self.cutmix,
                cutmix_minmax=self.cutmix_minmax,
                prob=self.mixup_prob,
                switch_prob=self.mixup_switch_prob,
                mode=self.mixup_mode,
                label_smoothing=self.smoothing,
                num_classes=self.num_classes,
            )

        self.criterion = torch.nn.CrossEntropyLoss()

        # add criterion
        if mixup_active:
            # smoothing is handled with mixup label transform
            self.criterion = SoftTargetCrossEntropy()
        elif self.smoothing > 0.:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=self.smoothing)

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
            num_enc_channels = self.fourier_enc_bands * len(self.index_dims)

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
        skip_list: List[str],
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
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay_and_layer_adaptation(
                self.named_parameters(),
                weight_decay=self.weight_decay,
                skip_list=['bias', 'bn', 'layer_norm'],
            )
        else:
            params = self.parameters()

        if self.optimizer == 'lamb':
            optimizer = LAMB(
                params, lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                params, lr=self.learning_rate, weight_decay=self.weight_decay
            )

        total_steps = self.train_iters_per_epoch * self.training_epochs

        if self.scheduler == 'no_warmup':
            decay_after_steps = self.train_iters_per_epoch * self.decay_start_from
            scheduler_fn = no_warmup_cosine_decay(decay_after_steps, total_steps)
        elif self.scheduler == 'warmup':
            warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
            scheduler_fn = linear_warmup_decay(warmup_steps, total_steps, cosine=True)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, scheduler_fn
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


    def shared_step(self, batch, is_training):
        x, y = batch
        logits = self.model(x, is_training)
        loss = self.criterion(logits, y)

        return loss, logits, y

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, y = self.shared_step(batch, is_training=True)
        acc = self.train_acc(F.softmax(logits, dim=1), y)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        self.log('train_acc_epoch', self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, y = self.shared_step(batch, is_training=False)
        self.val_acc(F.softmax(logits, dim=1), y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, y = self.shared_step(batch, is_training=False)
        self.test_acc(F.softmax(logits, dim=1), y)

        self.log('test_loss', loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

        return loss

if __name__ == '__main__':
    # TODO: add fast imagenet dataloading
    # TODO: do we need RASampler?
    # TODO: add args for transforms
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument('--local_rank', type=int, default=0)  # added to launch 2 ddp script on same node
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--data_dir', type=str,  default='')

    #  training params
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--num_accelerators', type=int, default=8)
    parser.add_argument('--strategy', type=str, default='ddp')

    parser.add_argument("--fp16", action="store_true")
    parser.set_defaults(fp16=False)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default='lamb', help='lamb/adamw')
    parser.add_argument('--scheduler', type=str, default='no_warmup', help='no_warmup/warmup')
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--gradient_clip_val', type=float, default=10.)

    parser.add_argument('--exclude_bn_bias', action="store_true")
    parser.set_defaults(exclude_bn_bias=True)

    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=str, default=0.1)
    parser.add_argument('--training_epochs', type=int, default=110)
    parser.add_argument('--decay_start_from', type=int, default=55)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # dataset params
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=224)

    # perceiver params
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--z_index_dim', type=int, default=512)
    parser.add_argument('--num_z_channels', type=int, default=1024)

    parser.add_argument('--num_self_attends_per_block', type=int, default=6)
    parser.add_argument('--num_self_attend_heads', type=int, default=8)
    parser.add_argument('--self_attend_widening_factor', type=int, default=1)

    parser.add_argument('--cross_attend_widening_factor', type=int, default=1)
    parser.add_argument('--num_cross_attend_heads', type=int, default=1)
    parser.add_argument('--cross_attention_shape_for_attn', type=str, default='kv')

    parser.add_argument('--use_query_residual', action="store_true")
    parser.set_defaults(use_query_residual=True)

    # preprocessor params
    parser.add_argument('--prep_type', type=str, default='pixels')
    parser.add_argument('--spatial_downsample', type=int, default=1)
    parser.add_argument('--conv1x1_num_img_channels', type=int, default=256)
    parser.add_argument('--concat_or_add_pos', type=str, default='concat')

    # pos encoding params
    parser.add_argument('--position_encoding_type', type=str, default='fourier')
    parser.add_argument('--trainable_enc_channels', type=int, default=256)
    parser.add_argument('--trainable_enc_init_scale', type=float, default=0.02)
    parser.add_argument('--fourier_enc_bands', type=int, default=64)

    parser.add_argument('--fourier_enc_concat_pos', action="store_true")
    parser.set_defaults(fourier_enc_concat_pos=True)

    # TODO: verify this is false and num_channels in input is 261
    parser.add_argument('--fourier_enc_sine_only', action="store_true")
    parser.set_defaults(fourier_enc_sine_only=False)

    # mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # initialize datamodule
    dm = ImagenetDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.img_size,
    )

    args.num_samples = dm.num_samples
    normalization = imagenet_normalization()

    # initialize transforms
    # defaults set to DeiT
    # TODO: add args
    dm.train_transforms = TrainTransform()
    dm.val_transforms = EvalTransform()
    dm.test_transforms = EvalTransform()

    # model init, checkpointing
    if args.ckpt_path == "":
        model = ImageClassification(**args.__dict__)
    else:
        print('____________________________________________________')
        print('Loading model weights from checkpoint...')
        print('Args initialized according to this script')
        print('____________________________________________________')
        model = ImageClassification.load_from_checkpoint(args.ckpt_path, strict=False, **args.__dict__)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(save_last=True),
    ]

    # trainer
    trainer = pl.Trainer(
        max_epochs=args.training_epochs,
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.num_accelerators,
        callbacks=callbacks,
        precision=16 if args.fp16 else 32,
        ckpt_path=None if args.ckpt_path == "" else args.ckpt_path,
        gradient_clip_val=args.gradient_clip_val,
    )
    trainer.fit(model, dm)
