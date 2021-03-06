import random
import numpy as np
from itertools import cycle
from utils import imshow_grid, transform_config

from torchvision import datasets
from torch.utils.data import Dataset, DataLoader


class MNIST_Paired(Dataset):
    def __init__(self, root='mnist', download=True, train=True, transform=transform_config):
        self.mnist = datasets.MNIST(root=root, download=download, train=train, transform=transform)

        self.data_dict = {}

        for i in range(self.__len__()):
            image, label = self.mnist.__getitem__(i)

            try:
                self.data_dict[label]
            except KeyError:
                self.data_dict[label] = []

            self.data_dict[label].append(image)

    def __len__(self):
        return self.mnist.__len__()

    def __getitem__(self, index):
        image, label = self.mnist.__getitem__(index)

        # return another image of the same class randomly selected from the data dictionary
        # this is done to simulate pair-wise labeling of data
        return image, random.SystemRandom().choice(self.data_dict[label]), label

class CIFAR_Paired(Dataset):
    def __init__(self, root='cifar', download=True, train=True, transform=transform_config):
        self.cifar = datasets.CIFAR10(root=root, download=download, train=train, transform=transform)

        self.data_dict = {}

        for i in range(self.__len__()):
            image, label = self.cifar.__getitem__(i)

            try:
                self.data_dict[label]
            except KeyError:
                self.data_dict[label] = []

            self.data_dict[label].append(image)

    def __len__(self):
        return self.cifar.__len__()

    def __getitem__(self, index):
        image, label = self.cifar.__getitem__(index)

        # return another image of the same class randomly selected from the data dictionary
        # this is done to simulate pair-wise labeling of data
        return image, random.SystemRandom().choice(self.data_dict[label]), label


if __name__ == '__main__':
    """
    test code for data loader
    """
    cifar_paired = CIFAR_Paired()
    loader = cycle(DataLoader(cifar_paired, batch_size=16, shuffle=True, num_workers=0, drop_last=True))

    print(cifar_paired.data_dict.keys())

    image_batch, image_batch_2, labels_batch = next(loader)
    print(labels_batch)

    image_batch = np.transpose(image_batch, (0, 2, 3, 1))
    # image_batch = np.concatenate((image_batch, image_batch, image_batch), axis=3)
    imshow_grid(image_batch)

    image_batch_2 = np.transpose(image_batch_2, (0, 2, 3, 1))
    # image_batch_2 = np.concatenate((image_batch_2, image_batch_2, image_batch_2), axis=3)
    imshow_grid(image_batch_2)
