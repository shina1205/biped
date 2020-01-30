# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def make_datapath_list(rootpath):
    datapath_list = []
    for data in os.listdir(rootpath):
        datapath = rootpath + str(data)
        datapath_list.append(datapath)

    return datapath_list


class PictureTransform():

    def __init__(self, size):
        self.data_transform = transforms.Compose([
            transforms.CenterCrop(512),
            transforms.Resize(size),
            transforms.ToTensor()])

    def __call__(self, picture):
        return self.data_transform(picture)


def mean_std(datapath_list, transform):
    tensor_list = []
    for datapath in datapath_list:
        picture = Image.open(datapath)
        tensor = transform(picture)
        tensor_list.append(tensor)

    tensors = torch.stack(tensor_list)

    mean = tensors.mean(dim=(0, 2, 3))
    mean = list(mean.numpy())
    std = tensors.std(dim=(0, 2, 3))
    std = list(std.numpy())

    return mean, std


rootpath = './video/1min/'
datapath_list = make_datapath_list(rootpath)

size = 256
mean, std = mean_std(datapath_list, transform=PictureTransform(size))
print('mean: {}\nstd: {}'.format(mean, std))
