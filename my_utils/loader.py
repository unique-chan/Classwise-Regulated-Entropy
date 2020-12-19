import os
from torch.utils import data
from torchvision import datasets
import torchvision.transforms as transforms


class Loader:
    def __init__(self, dataset_path):
        self.train_dir = os.path.join(dataset_path, 'train')
        self.valid_dir = os.path.join(dataset_path, 'valid')
        self.test_dir = os.path.join(dataset_path, 'test')
        self.num_classes = Loader.__get_num_classes(self.train_dir)

    @staticmethod
    def __get_num_classes(root_dir):
        return len([dir_ for dir_ in os.listdir(root_dir) if not os.path.isfile(dir_)])

    @staticmethod
    def __get_train_transform(image_height, image_width):
        transforms_list = [
            transforms.RandomCrop((image_height, image_width), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        return transforms_list

    def get_train_loader(self, image_height, image_width, batch_size, shuffle=True, num_workers=0):
        transforms_list = Loader.__get_train_transform(image_height, image_width)
        train_set = datasets.ImageFolder(root=self.train_dir, transform=transforms.Compose(transforms_list))
        return data.DataLoader(train_set, batch_size=batch_size,
                               shuffle=shuffle, num_workers=num_workers)

    def get_valid_loader(self, batch_size, shuffle=False, num_workers=0):
        valid_set = datasets.ImageFolder(root=self.valid_dir, transform=transforms.Compose([transforms.ToTensor()]))
        return data.DataLoader(valid_set, batch_size=batch_size,
                               shuffle=shuffle, num_workers=num_workers)

    def get_test_loader(self, batch_size, shuffle=False, num_workers=0):
        test_set = datasets.ImageFolder(root=self.test_dir, transform=transforms.Compose([transforms.ToTensor()]))
        return data.DataLoader(test_set, batch_size=batch_size,
                               shuffle=shuffle, num_workers=num_workers)
