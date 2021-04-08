'''
Author : ZuoZuo
Time: 2021.4.
'''


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from PIL import Image
import glob
from model.model import *
import numpy as np

device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
print(device)

trainroot = r"C:\Users\JQRZJG\Desktop\目标检测网络\datasets\CIFAR\data\trainset"
trainset = datasets.CIFAR10(trainroot, train=True, transform=None, target_transform=None, download=True)

valroot = r"C:\Users\JQRZJG\Desktop\目标检测网络\datasets\CIFAR\data\valset"
valset = datasets.CIFAR10(valroot, train=False, transform=None, target_transform=None, download=True)

trainset_size = len(trainset)
valset_size = len(valset)

class CIFARDatasets(Dataset):

    def __init__(self, dataset_type="train", transform=None):
        self.transform = transform
        self.dataset_type = dataset_type
        self.traindata, self.validationdata = self.load_dataset()

    def __getitem__(self, index):
        if self.dataset_type == "train":
            img, target = self.traindata[index][0], self.traindata[index][1]
            if self.transform is not None:
                img = self.transform(img)
                print(type(img))
                #target = self.transform(target)
        if self.dataset_type == "val":
            img, target = self.traindata[index][0], self.traindata[index][1]
            if self.transform is not None:
                img = self.transform(img)
                target = self.transform(target)
        return img, target


    def __len__(self):
        if self.dataset_type == "train":
            return trainset_size
        if self.dataset_type == "val":
            return valset_size

    def load_dataset(self,):

        train_data = []
        val_data = []
        k = []
        for i in range(trainset_size):
            k.append(trainset[i][0])
            k.append(trainset[i][1])
            train_data.append(k)
            k = []
        for i in range(valset_size):
            k.append(valset[i][0])
            k.append(self.vectorized_label(valset[i][1]))
            val_data.append(k)
            k = []
        return train_data, val_data


    def vectorized_label(self, j):
        e = np.zeros((10))
        e[j] = 1.0

        return e


start_step = 0
end_step = 2


def Train():
    Loss = []
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    epoch = 20
    dataset = CIFARDatasets(transform= data_transform["train"])
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    net = resnet101(10)
    criterion = nn.CrossEntropyLoss()       #定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.0001, momentum=0.9)  #定义优化器

    net = resnet101(10)

    save_path = r"C:\Users\JQRZJG\Desktop\object detection Network\Resnet\resnet\backup\resNet.pth"
    for epoch in range(epoch):
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            image, label = data
            optimizer.zero_grad()

            output = net(image)
            print(label.shape)
            loss = criterion(output, label.long())
            print(output)
            print(label)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            Loss.append(loss)
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
        Loss.append(running_loss)



if __name__ == "__main__":
    Train()
    print(Loss)








