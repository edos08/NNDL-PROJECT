import os
import random
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import seaborn as sns

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import to_tensor
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def show_images_with_text(images: list, labels: list = None, title: str = None, figsize: tuple = None):
    """
    Display an array of images from a list of images.

    Args:
        images (list): List of images to display
        labels (list): List of labels to display
        title (str): Title of the image
        figsize (tuple): Figure size
    """
    # create a new array with a size large enough to contain all the images

    if figsize is None:
        figsize = (20.0, 18.0)

    dim = math.ceil(math.sqrt(len(images)))

    plt.figure(figsize=figsize)

    for i, image, label in zip(range(len(images)), images, labels):
        ax = plt.subplot(dim, dim, i + 1)
        plt.imshow(image)
        plt.axis("off")
        if label is not None:
            ax.text(s=label, y=160, x=320, fontsize=14, ha="center")

    if title is not None:
        plt.suptitle(title, fontsize=20)

    plt.show()


def plot_img(image: np.ndarray, title: str = "Image"):
    """
    Plot an image.

    Args:
        image (np.ndarray): image to plot
        title (str): title of the image
    """

    # Convert the image from BGR to RGB
    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(img_rgb)
    plt.title(title)
    plt.show()
    plt.close()


def random_color_generator():
    """
    Generate a random color in hexadecimal.
    """

    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    color = "#%02x%02x%02x" % (r, g, b)

    return color


def show_plot(signals: list, title: str, x_label: str, y_label: str, x_range: list = None, y_range: list = None):
    """
    Plot a graph of signal

    Args:
        signals (list): list of signals
        title (str): title of the graph
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        x_range (list): range of the x-axis
        y_range (list): range for the y-axis
    """
    ax = sns.lineplot(x=signals[0][0], y=signals[0][1], lw=3, color=random_color_generator(), label=signals[0][2])
    for signal in signals[1:]:
        sns.lineplot(x=signal[0], y=signal[1], lw=3, color=random_color_generator(), label=signal[2])

    ax.figure.set_size_inches(20, 8)
    ax.set_title(title, fontsize=22)
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.legend(loc="best")
    if x_range is not None:
        ax.set_xlim(*x_range)
    if y_range is not None:
        ax.set_ylim(*y_range)

    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()


def compute_mean_std_from_dataset(data: Dataset) -> tuple:
    # see https://gist.github.com/spirosdim/79fc88231fffec347f1ad5d14a36b5a8
    """
    Compute the mean and standard deviation from a list of images
    We assume that the images of the dataloader have the same height and width
    
    Args:
        data (Dataset): Dataset from which to compute the mean and standard deviation
    """
    channels_sum, channels_sqr_sum = torch.zeros(3), torch.zeros(3)
    num_batches = 0

    for image, _ in data:  # shape of images: [c,w,h]
        image = to_tensor(image)
        channels_sum += torch.mean(image, dim=[1, 2])
        channels_sqr_sum += torch.mean(image ** 2, dim=[1, 2])
        num_batches += 1

    mean = channels_sum / num_batches
    std = torch.sqrt(channels_sqr_sum / num_batches - mean ** 2)

    return mean, std


def compute_mean_std(loader: DataLoader) -> tuple:
    # see https://gist.github.com/spirosdim/79fc88231fffec347f1ad5d14a36b5a8
    """
    Compute the mean and standard deviation from a list of images
    We assume that the images of the dataloader have the same height and width
    
    Args:
        loader (DataLoader): Dataset loader to compute the mean and standard deviation
    """
    channels_sum, channels_sqr_sum = torch.zeros(3), torch.zeros(3)
    num_batches = 0

    for batch_images, _ in tqdm(loader):  # shape of images: [b,c,w,h]
        channels_sum += torch.mean(batch_images, dim=[0, 2, 3])
        channels_sqr_sum += torch.mean(batch_images ** 2, dim=[0, 1, 2])
        num_batches += 1

    mean = channels_sum / num_batches
    std = torch.sqrt(channels_sqr_sum / num_batches - mean ** 2)

    return mean, std


def compute_mean_std_from_images(images: list) -> tuple:
    """
    Compute the mean and standard deviation from a list of images

    Args:
        images (list): Dataset loader to compute the mean and standard deviation
    """

    mean = np.zeros(3)
    std = np.zeros(3)

    for image in images:
        average_color_row = np.average(image, axis=0)
        average_color = np.average(average_color_row, axis=0)
        mean = mean + average_color

    for image in images:
        for i in range(3):
            std[i] = std[i] + ((image[:, :, i] - mean[i]) ** 2).sum() / (image.shape[0] * image.shape[1])

    mean = mean / len(images)
    std = np.sqrt(std / len(images))

    return mean, std


def train_val_dataset(dataset: Dataset, val_split: float = 0.2) -> dict[str, Subset[Any]]:
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}
    return datasets


def show_grid(images, labels):
    out = torchvision.utils.make_grid(images)
    np_img = out.numpy()
    np.clip(np_img, 0, 1)
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
    # TO BE FIXED
    # print(' '.join(f'{label_dict[labels[j].item()]}' for j in range(len(labels))))


def read_images(path: str) -> list:
    """
    Read images from a parent path and return a list of images

    Args:
        path (str): path to images
    """
    data = list()

    directories = list()

    for dir_name, _, _ in os.walk(path):
        directories.append(dir_name)

    for dir_name, _, filenames in os.walk(path):
        if len(directories) > 1:
            for filename in filenames:
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    label = dir_name.split("/")[-1]
                    image = os.path.join(dir_name, filename)
                    data.append((label, image))
        else:
            for filename in filenames:
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    label = filename.split("_")[0]
                    image = os.path.join(dir_name, filename)
                    data.append((label, image))

    return data


class ImagesLabelsDataset(Dataset):
    """
    Custom dataset for loading images and labels from a list of images and labels
    """

    def __init__(self, images_array, labels_array, transforms=None):
        self.labels = labels_array
        self.images = images_array
        self.transforms = transforms

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.labels)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        print(self.counter)
        print(validation_loss)
        if validation_loss + self.min_delta < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
