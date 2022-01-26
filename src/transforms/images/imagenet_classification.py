# adapted from: https://github.com/facebookresearch/deit/blob/main/datasets.py
import torchvision.transforms as T
from timm.data import create_transform
from src.datamodules.images.imagenet import imagenet_normalization


class TrainTransform:
    """Image classification train transform.

    Args:
        img_size:
        color_jitter:
        aa: Use AutoAugment policy. "v0" or "original".
            (default: rand-m9-mstd0.5-inc1)
        interpolation:
        reprob: Random erase prob (default: 0.25)
        remode: Random erase mode (default: "pixel")
        recount: Random erase count (default: 1)
    """

    def __init__(
        self,
        img_size=224,
        color_jitter=0.4,
        aa='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        reprob=0.25,
        remode='pixel',
        recount=1,
    ) -> None:
        # this should always dispatch to transforms_imagenet_train in timm
        self.transform = create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation=interpolation,
            re_prob=reprob,
            re_mode=remode,
            re_count=recount,
        )

    def __call__(self, x):
        return self.transform(x)


class EvalTransform:
    def __init__(
        self,
        img_size=224,
        normalize=imagenet_normalization(),
    ) -> None:

        size = int((256 / 224) * img_size)

        self.transform = [
            T.Resize(size, interpolation=3),
            T.CenterCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]

        if normalize is not None:
            self.transform.append(normalize)

        self.transform = T.Compose(self.transform)

    def __call__(self, x):
        return self.transform(x)
