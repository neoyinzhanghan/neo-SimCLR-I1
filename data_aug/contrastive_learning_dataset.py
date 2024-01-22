from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection

import os
from wsissl.dataset import PFDataset, folder_train_test_split


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.ToTensor(),
            ]
        )
        return data_transforms

    def get_dataset(self, name, n_views, num_images_per_epoch):
        valid_datasets = {
            "cifar10": lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(32), n_views
                ),
                download=True,
            ),
            "stl10": lambda: datasets.STL10(
                self.root_folder,
                split="unlabeled",
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96), n_views
                ),
                download=True,
            ),
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            if os.path.isdir(name):
                # train_folders, _ = folder_train_test_split(
                #     os.path.join(name, "train"), train_prop=1
                # )
                # train_dataset = PFDataset(
                #     folders=train_folders,
                #     num_images_per_epoch=num_images_per_epoch,  # args.num_images_per_epoch,
                #     transform=ContrastiveLearningViewGenerator(
                #         self.get_simclr_pipeline_transform(96), n_views
                #     ),
                # )
                # print(train_dataset)

                # name is a path to an imagenet like dataset
                train_dataset = datasets.ImageFolder(
                    name,
                    transform=ContrastiveLearningViewGenerator(
                        self.get_simclr_pipeline_transform(96), n_views
                    ),
                )

                return train_dataset
            else:
                raise InvalidDatasetSelection()
        else:
            return dataset_fn()
