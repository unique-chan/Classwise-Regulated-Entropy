import os
import torch
from torch.utils import data
from torchvision import datasets
import torchvision.transforms as transforms


class Loader:
    def __init__(self, dataset_path, image_height, image_width, batch_size=128, num_workers=0):
        self.train_dir = os.path.join(dataset_path, 'train')
        self.valid_dir = os.path.join(dataset_path, 'valid')
        self.test_dir = os.path.join(dataset_path, 'test')
        self.num_workers = num_workers
        self.num_classes = Loader.__get_num_classes(self.train_dir)
        self.image_height, self.image_width = image_height, image_width
        self.batch_size = batch_size
        self.train_mean, self.train_std = self.get_train_mean_std()

    def get_train_mean_std(self):
        loader = self.get_train_loader_for_mean_std()
        mean, std = torch.zeros(3), torch.zeros(3)
        for inputs, targets in loader:
            for i in range(3):
                mean[i] += inputs[:, i, :, :].mean()
                std[i] += inputs[:, i, :, :].std()
        mean /= len(loader)
        std /= len(loader)
        return mean, std

    @staticmethod
    def __get_num_classes(root_dir):
        return len([dir_ for dir_ in os.listdir(root_dir) if not os.path.isfile(dir_)])

    @staticmethod
    def __get_train_transform(image_height, image_width, mean, std):
        transforms_list = [
            transforms.RandomCrop((image_height, image_width), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),   # [HxWxC] [0, 255] -> [CxHxW] [0., 1.]
            transforms.Normalize(mean, std)
        ]
        return transforms.Compose(transforms_list)

    @staticmethod
    def __get_eval_transform(mean, std):
        transforms_list = [
            transforms.ToTensor(),   # [HxWxC] [0, 255] -> [CxHxW] [0., 1.]
            transforms.Normalize(mean, std)
        ]
        return transforms.Compose(transforms_list)

    def get_train_loader_for_mean_std(self):
        train_set = datasets.ImageFolder(root=self.train_dir, transform=transforms.ToTensor())
        return data.DataLoader(train_set, batch_size=self.batch_size,
                               shuffle=False, num_workers=self.num_workers)

    def get_train_loader(self, shuffle=True):
        composed_transforms_list = Loader.__get_train_transform(self.image_height, self.image_width,
                                                                self.train_mean, self.train_std)
        train_set = datasets.ImageFolder(root=self.train_dir, transform=composed_transforms_list)
        return data.DataLoader(train_set, batch_size=self.batch_size,
                               shuffle=shuffle, num_workers=self.num_workers)

    def get_valid_loader(self, shuffle=False):
        valid_set = datasets.ImageFolder(root=self.valid_dir,
                                         transform=Loader.__get_eval_transform(self.train_mean, self.train_std))
        return data.DataLoader(valid_set, batch_size=self.batch_size,
                               shuffle=shuffle, num_workers=self.num_workers)

    def get_test_loader(self, shuffle=False):
        test_set = datasets.ImageFolder(root=self.test_dir,
                                        transform=Loader.__get_eval_transform(self.train_mean, self.train_std))
        return data.DataLoader(test_set, batch_size=self.batch_size,
                               shuffle=shuffle, num_workers=self.num_workers)
