import random
import numpy as np
from itertools import cycle
from utils import imshow_grid, transform_config

from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

class DSPRITES_Paired(Dataset):
  def __init__(self, train=True, transform=transform_config):

    # Load and unpack data
    dsprites_zip = np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='latin1')
    self.imgs = dsprites_zip['imgs'].reshape((-1, 1, 64, 64))
    self.meta = dsprites_zip['metadata'][()]
    self.latent_values = dsprites_zip['latents_values']
    self.latent_classes = dsprites_zip['latents_classes']

  def __len__(self):
    return len(self.imgs) // 2

  def __getitem__(self, idx):
    latents_sizes = self.meta['latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1,])))

    # keeping the first four latents fixed
    a = np.random.choice(latents_sizes[0])
    b = np.random.choice(latents_sizes[1])
    c = np.random.choice(latents_sizes[2])
    d = np.random.choice(latents_sizes[3])

    # x and y are sampled randomly for each sample
    x_1 = np.random.choice(latents_sizes[4])
    x_2 = np.random.choice(latents_sizes[4])
    y_1 = np.random.choice(latents_sizes[5])
    y_2 = np.random.choice(latents_sizes[5])

    # assemble the two latent vectors
    latent_1 = [a, b, c, d, x_1, y_1]
    latent_2 = [a, b, c, d, x_2, y_2]

    # grab the two images
    img_idx_1 = np.dot(latent_1, latents_bases).astype(int)
    img_idx_2 = np.dot(latent_2, latents_bases).astype(int)
    img_1 = self.imgs[img_idx_1]
    img_2 = self.imgs[img_idx_2]

    return img_1, img_2, latent_1, latent_2

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
