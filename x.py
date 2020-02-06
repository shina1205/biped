# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


class PictureTransform():

    def __init__(self, size, mean, std):
        self.data_transform = transforms.Compose([
            transforms.CenterCrop(512),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    def __call__(self, picture):
        return self.data_transform(picture)


# size = 256
size = 64
mean = (0.293,)
std = (0.283,)
transform = PictureTransform(size, mean, std)

picture_num = 300
datapath = './video/1min/{}.jpg'.format(picture_num)
picture = Image.open(datapath)
x = transform(picture)

x = x[0].cpu().detach().numpy()
plt.imshow(x, 'gray')
plt.savefig('x.jpg')
# plt.show()
