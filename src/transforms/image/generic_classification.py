# adapted from: https://github.com/facebookresearch/deit/blob/main/datasets.py
import torchvision.transforms as T
from src.datamodules.image.imagenet import imagenet_normalization


class GenericTrainTransform:
    """Image classification train transform.
    """

    def __init__(
        self,
        img_size=224,
        color_jitter=1.,
        gaussian_blur=True,
        normalize=None,
    ) -> None:
        self.color_jitter = T.ColorJitter(
            0.8 * color_jitter,
            0.8 * color_jitter,
            0.8 * color_jitter,
            0.2 * color_jitter,
        )

        data_transforms = [
            T.RandomResizedCrop(size=img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([self.color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
        ]

        if gaussian_blur:
            kernel_size = int(0.1 * img_size)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(
                T.RandomApply([T.GaussianBlur(kernel_size=kernel_size)], p=0.5)
            )

        data_transforms = T.Compose(data_transforms)

        if normalize is None:
            self.final_transform = T.ToTensor()
        else:
            self.final_transform = T.Compose(
                [T.ToTensor(), normalize]
            )

        self.transform = T.Compose([data_transforms, self.final_transform])

    def __call__(self, x):
        return self.transform(x)


class GenericEvalTransform:
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
