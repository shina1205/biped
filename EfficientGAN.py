# -*- coding: utf-8 -*-
import random
import math
import time
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sync_batchnorm import convert_model
from collections import OrderedDict


def make_datapath_list(rootpath):
    datapath_list = []
    for data in os.listdir(rootpath):
        datapath = rootpath + str(data)
        datapath_list.append(datapath)

    return datapath_list


class PictureTransform():

    def __init__(self, size, mean, std):
        self.data_transform = transforms.Compose([
            transforms.CenterCrop(512),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    def __call__(self, picture):
        return self.data_transform(picture)


class GAN_Dataset(data.Dataset):

    def __init__(self, datapath_list, transform):
        self.datapath_list = datapath_list
        self.transform = transform

    def __len__(self):
        return len(self.datapath_list)

    def __getitem__(self, index):
        picture_transformed = self.pull_item(index)

        return picture_transformed

    def pull_item(self, index):
        datapath = self.datapath_list[index]
        picture = Image.open(datapath)
        picture_transformed = self.transform(picture)

        return picture_transformed


class Discriminator(nn.Module):

    def __init__(self, z_dim=100):
        super(Discriminator, self).__init__()

        self.x_layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.5))

        self.x_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.5))

        self.x_layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.5))

        self.x_layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.5))

        self.z_layer1 = nn.Linear(z_dim, 512)

        self.last1 = nn.Sequential(
            nn.Linear(512 * 4 * 4 + 512, 1024),
            nn.LeakyReLU(0.2, inplace=True))

        self.last2 = nn.Linear(1024, 1)

    def forward(self, x, z):
        x_out = self.x_layer1(x)
        x_out = self.x_layer2(x_out)
        x_out = self.x_layer3(x_out)
        x_out = self.x_layer4(x_out)

        z = z.view(z.shape[0], -1)
        z_out = self.z_layer1(z)

        x_out = x_out.view(-1, 512 * 4 * 4)
        out = torch.cat([x_out, z_out], dim=1)
        out = self.last1(out)

        feature = out
        feature = feature.view(feature.size()[0], -1)

        out = self.last2(out)

        return out, feature


class Generator(nn.Module):

    def __init__(self, z_dim=100):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=512,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.last = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=1,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)

        return out


class Encoder(nn.Module):

    def __init__(self, z_dim=100):
        super(Encoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True))

        self.last = nn.Linear(512 * 8 * 8, z_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, 512 * 8 * 8)
        out = self.last(out)

        return out


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def train_model(D, G, E, dataloaders_dict, num_epochs):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_ids = [0, 1, 2, 3]
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(device_ids[0]))
    else:
        device = torch.device('cpu')
    print('device: ', device)

    lr_d = 0.0002
    lr_g = 0.0002
    lr_e = 0.0002
    beta1, beta2 = 0.5, 0.999
    d_optimizer = torch.optim.Adam(D.parameters(), lr_d, [beta1, beta2])
    g_optimizer = torch.optim.Adam(G.parameters(), lr_g, [beta1, beta2])
    e_optimizer = torch.optim.Adam(E.parameters(), lr_e, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    z_dim = 100
    mini_batch_size = 256

    D = convert_model(nn.DataParallel(D))
    G = convert_model(nn.DataParallel(G))
    E = convert_model(nn.DataParallel(E))

    D.to(device)
    G.to(device)
    E.to(device)

    D.train()
    G.train()
    E.train()

    torch.backends.cudnn.benchmark = True

    num_train_list = len(dataloaders_dict['train'].dataset)
    batch_size = dataloaders_dict['train'].batch_size

    iteration = 1
    logs = []

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_e_loss = 0.0

        # print('-------------')
        # print('epoch: {}/{}'.format(epoch + 1, num_epochs))
        # print('-------------')
        # print(' (train) ')

        for x in dataloaders_dict['train']:
            if x.size()[0] == 1:
                continue

            mini_batch_size = x.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            x = x.to(device)

            # --------------------
            # 1. Discriminator
            # --------------------
            z_out_real = E(x)
            d_out_real, _ = D(x, z_out_real)

            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            fake_x = G(input_z)
            d_out_fake, _ = D(fake_x, input_z)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # --------------------
            # 2. Generator
            # --------------------
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            fake_x = G(input_z)
            d_out_fake, _ = D(fake_x, input_z)

            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # --------------------
            # 3. Encoder
            # --------------------
            z_out_real = E(x)
            d_out_real, _ = D(x, z_out_real)

            e_loss = criterion(d_out_real.view(-1), label_fake)

            e_optimizer.zero_grad()
            e_loss.backward()
            e_optimizer.step()

            # --------------------
            # 4. Record
            # --------------------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_e_loss += e_loss.item()
            iteration += 1

        t_epoch_finish = time.time()
        print('-------------')
        print('epoch: {} || D_loss: {:.4f} || G_loss: {:.4f} || E_loss: {:.4f}'.format(
            epoch + 1, epoch_d_loss / batch_size, epoch_g_loss / batch_size, epoch_e_loss / batch_size))
        print('timer: {:.1f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {
            'epoch': epoch + 1,
            'D_loss': epoch_d_loss / batch_size,
            'G_loss': epoch_g_loss / batch_size,
            'E_loss': epoch_e_loss / batch_size}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv('log_GAN.csv')

    torch.save(D.state_dict(), 'weight/D_sync.pth')
    torch.save(G.state_dict(), 'weight/G_sync.pth')
    torch.save(E.state_dict(), 'weight/E_sync.pth')

    return D, G, E


def multi_state_dict(weight, map_location={'cuda:0', 'cpu'}):
    state_dict = torch.load(weight, map_location=map_location)
    multi_state_dict = OrderedDict()

    for i, j in state_dict.items():
        if 'module' in i:
            i = i.replace('module.', '')
        multi_state_dict[i] = j

    return multi_state_dict


def detection(x, fake_x, z_out_real, D, alpha=0.1):
    residual_loss = torch.abs(x - fake_x)
    residual_loss = residual_loss.view(residual_loss.size()[0], -1)
    residual_loss = torch.sum(residual_loss, dim=1)

    _, x_feature = D(x, z_out_real)
    _, fake_x_feature = D(fake_x, z_out_real)

    discrimination_loss = torch.abs(x_feature - fake_x_feature)
    discrimination_loss = discrimination_loss.view(
        discrimination_loss.size()[0], -1)
    discrimination_loss = torch.sum(discrimination_loss, dim=1)

    loss = (1 - alpha) * residual_loss + alpha * discrimination_loss

    total_loss = torch.sum(loss)

    return total_loss, loss, residual_loss


def colormap(x, fake_x, size):
    abs = np.abs(x - fake_x).reshape(size, size, 1)

    colormap = cv2.applyColorMap(np.uint8(255 * abs), cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    return colormap


torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

rootpath = './video/1min/'
picture_list = make_datapath_list(rootpath)

size = 64
mean = (0.293,)
std = (0.283,)
dataset = GAN_Dataset(datapath_list=picture_list,
                      transform=PictureTransform(size, mean, std))

train_size = 0.9
train_dataset, test_dataset = train_test_split(
    dataset, train_size=train_size, shuffle=False)

batch_size = 256
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {'train': train_dataloader, 'test': test_dataloader}

D = Discriminator(z_dim=100)
G = Generator(z_dim=100)
E = Encoder(z_dim=100)

# --------------------
# 1. Train
# --------------------
D.apply(weight_init)
G.apply(weight_init)
E.apply(weight_init)

num_epochs = 4
D_update, G_update, E_update = train_model(
    D, G, E, dataloaders_dict=dataloaders_dict, num_epochs=num_epochs)

# --------------------
# 2. Test
# --------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device_ids = [0, 1, 2, 3]
# if torch.cuda.is_available():
#     device = torch.device('cuda:{}'.format(device_ids[0]))
# else:
#     device = torch.device('cpu')

D_state_dict = multi_state_dict('./weight/D_sync.pth',
                                map_location={'cuda:0': 'cpu'})
G_state_dict = multi_state_dict('./weight/G_sync.pth',
                                map_location={'cuda:0': 'cpu'})
E_state_dict = multi_state_dict('./weight/E_sync.pth',
                                map_location={'cuda:0': 'cpu'})

D.load_state_dict(D_state_dict)
G.load_state_dict(G_state_dict)
E.load_state_dict(E_state_dict)
print('Loaded learned weights.')

D.to(device)
G.to(device)
E.to(device)

D.eval()
G.eval()
E.eval()

z_dim = 100
fixed_z = torch.randn(batch_size, z_dim)
fake_x = G(fixed_z.to(device))

batch_iterator = iter(dataloaders_dict['test'])
pictures = next(batch_iterator)

x = pictures[0:5]
x = x.to(device)

z_out_real = E(x)
fake_x = G(z_out_real)

total_loss, loss, residual_loss = detection(
    x, fake_x, z_out_real, D, alpha=0.1)

loss = loss.cpu().detach().numpy()
print('total loss: {}'.format(np.round(loss, 0)))

fig = plt.figure(figsize=(15, 5))
for i in range(5):
    _x = x[i][0].cpu().detach().numpy()
    _fake_x = fake_x[i][0].cpu().detach().numpy()
    heatmap = colormap(_x, _fake_x, size)

    plt.subplot(3, 5, i + 1)
    plt.imshow(_x, 'gray')

    plt.subplot(3, 5, 5 + i + 1)
    plt.imshow(_fake_x, 'gray')

    plt.subplot(3, 5, 10 + i + 1)
    plt.imshow(heatmap)

# plt.show()
plt.savefig('heatmap.png')
print('Save complete.')
