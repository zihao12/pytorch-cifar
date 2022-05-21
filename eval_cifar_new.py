## evaluate trained model on cifar10.1 data
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import math
import os
import random
import sys
import pdb

# repo_root = os.path.join(os.getcwd(), '../code')
# sys.path.append(repo_root)

# script_dir = "../"
# sys.path.append(os.path.abspath(script_dir))

from IPython.display import display
from ipywidgets import Layout
from ipywidgets import widgets
from matplotlib import pyplot as plt
import numpy as np
import tqdm

import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
# import cifar10_1

from models import *
from cifar_new import CIFAR_New

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## lodaing data
print("\nloading data")
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = CIFAR_New(root='./data', transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)



# cifar_label_names = cifar10_1.cifar10_label_names
# version = 'v6'
# images, labels = cifar10_1.load_new_test_data(version)
# num_images = images.shape[0]
# print('\nLoaded version "{}" of the CIFAR-10.1 dataset.'.format(version))
# print('There are {} images in the dataset.'.format(num_images))

# print("\nconstruct data loader")
# images = torch.Tensor(images)
# labels = torch.Tensor(labels)
# dataset = torch.utils.data.TensorDataset(images, labels)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000)

print("\nload trained model")
net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']

print("compute test accuracy")
criterion = nn.CrossEntropyLoss()
net.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

acc = 100. * correct/total
print(f"\nprev test accuracy {best_acc}")
print(f"curr test accuracy {acc}")

#pdb.set_trace()
