import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from src.datamodules.imagenet_dataset import CustomImagenet


def imagenet_normalization():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize


class ImagenetDataModule(LightningDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/
        Sample-of-Images-from-the-ImageNet-Dataset-used-in-the-ILSVRC-Challenge.png
        :width: 400
        :alt: Imagenet

    Specs:
        - 1000 classes
        - Each image is (3 x varies x varies) (here we default to 3 x 224 x 224)

    Imagenet train, val and test dataloaders.

    The train set is the imagenet train.

    The val set is taken from the train set with `num_imgs_per_val_class` images per class.
    For example if `num_imgs_per_val_class=2` then there will be 2,000 images in the validation set.

    The test set is the official imagenet validation set.

     Example::

        from pl_bolts.datamodules import ImagenetDataModule

        dm = ImagenetDataModule(IMAGENET_PATH)
        model = LitModel()

        Trainer().fit(model, dm)
    """

    name = 'imagenet'

    def __init__(
            self,
            data_dir: str,
            meta_dir: Optional[str] = None,
            num_imgs_per_val_class: int = 50,
            image_size: int = 224,
            num_workers: int = 16,
            batch_size: int = 32,
            *args,
            **kwargs,
    ):
        """
        Args:
            data_dir: path to the imagenet dataset file
            meta_dir: path to meta.bin file
            num_imgs_per_val_class: how many images per class for the validation set
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
        """
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.meta_dir = meta_dir
        self.num_imgs_per_val_class = num_imgs_per_val_class
        self.batch_size = batch_size
        self.num_samples = 1281167 - self.num_imgs_per_val_class * self.num_classes

    @property
    def num_classes(self):
        """
        Return:

            1000

        """
        return 1000

    def _verify_splits(self, data_dir, split):
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(f'a {split} Imagenet split was not found in {data_dir},'
                                    f' make sure the folder contains a subfolder named {split}')

    def prepare_data(self):
        """
        This method already assumes you have imagenet2012 downloaded.
        It validates the data using the meta.bin.

        .. warning:: Please download imagenet on your own first.
        """
        self._verify_splits(self.data_dir, 'train')
        self._verify_splits(self.data_dir, 'val')

        for split in ['train', 'val']:
            files = os.listdir(os.path.join(self.data_dir, split))
            if 'meta.bin' not in files:
                raise FileNotFoundError("""
                no meta.bin present. Imagenet is no longer automatically downloaded by PyTorch.
                To get imagenet:
                1. download yourself from http://www.image-net.org/challenges/LSVRC/2012/downloads
                2. download the devkit (ILSVRC2012_devkit_t12.tar.gz)
                3. generate the meta.bin file using the devkit
                4. copy the meta.bin file into both train and val split folders

                To generate the meta.bin do the following:

                from pl_bolts.datasets.imagenet_dataset import UnlabeledImagenet
                path = '/path/to/folder/with/ILSVRC2012_devkit_t12.tar.gz/'
                UnlabeledImagenet.generate_meta_bins(path)
                """)

    def train_dataloader(self):
        """
        Uses the train split of imagenet2012 and puts away a portion of it for the validation split
        """
        transforms = self.train_transform() if self.train_transforms is None else self.train_transforms

        dataset = CustomImagenet(
            self.data_dir,
            num_imgs_per_class=-1,
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            meta_dir=self.meta_dir,
            split='train',
            transform=transforms
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

        return loader

    def val_dataloader(self):
        """
        Uses the part of the train split of imagenet2012  that was not used for training via `num_imgs_per_val_class`

        Args:
            batch_size: the batch size
            transforms: the transforms
        """
        transforms = self.train_transform() if self.val_transforms is None else self.val_transforms

        dataset = CustomImagenet(
            self.data_dir,
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            meta_dir=self.meta_dir,
            split='val',
            transform=transforms
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return loader

    def test_dataloader(self):
        """
        Uses the validation split of imagenet2012 for testing
        """
        transforms = self.val_transform() if self.test_transforms is None else self.test_transforms

        dataset = CustomImagenet(
            self.data_dir,
            num_imgs_per_class=-1,
            meta_dir=self.meta_dir,
            split='test',
            transform=transforms
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )

        return loader

    def train_transform(self):
        preprocessing = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_normalization(),
        ])

        return preprocessing

    def val_transform(self):
        preprocessing = transforms.Compose([
            transforms.Resize(self.image_size + 32),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            imagenet_normalization(),
        ])

        return preprocessing
