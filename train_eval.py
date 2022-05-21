'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle
import numpy as np
import pdb

from models import *
from utils import progress_bar
from cifar_new import CIFAR_New



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--i', default=0, type=int, help='index experiments')
parser.add_argument('--run', default=10, type=int, help='number of experiments')
parser.add_argument('--maxiter', default=10, type=int, help='max number of iterations')


# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
args = parser.parse_args()

## set seed for reproducibility
torch.manual_seed(args.i)






# Training
def train(trainloader, net, device, criterion, optimizer, scheduler, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(testloader, net, device, criterion):
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return correct/total

def eval(testloader, net, device, set_name="test set"):
    net.eval()
    acc = np.array([])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            acc = np.append(acc, predicted.eq(targets).cpu().detach().numpy())

    print(f'acc on {set_name} is {acc.mean()}')
    return acc
            

def train_full(all_trainset, testloader, testloader_new, maxiter):
    indices = torch.randperm(len(all_trainset))[:int(0.7 * len(all_trainset))]
    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(all_trainset, indices), 
    batch_size=128, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Building model..')
    net = SimpleDLA()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    print('==> Start training..')
    for epoch in range(maxiter):
        train(trainloader, net, device, criterion, optimizer, scheduler, epoch)
        if epoch % 10 == 0:
            print("acc on test data")
            _ = test(testloader, net, device, criterion)
            print("acc on test-new data")
            _ = test(testloader_new, net, device, criterion)
        scheduler.step()

    return net, indices, device

def experiment(all_trainloader, all_trainset, testloader, testloader_new, outfile, maxiter = 10):
    net, indices, device = train_full(all_trainset, testloader, testloader_new, maxiter)
    print('==> Evaluate model..')
    out = {}
    out["indices"] = indices
    out["acc_train"] = eval(all_trainloader, net, device, set_name="all training set")
    out["acc_test"] = eval(testloader, net, device, set_name="test set")
    out["acc_test_new"] = eval(testloader_new, net, device, set_name="test-new set")

    with open(outfile, 'wb') as f:
        pickle.dump(out, f)

    


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

all_trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
all_trainloader = torch.utils.data.DataLoader(
    all_trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

testset_new = CIFAR_New(root='./data', transform=transform_test)
testloader_new = torch.utils.data.DataLoader(
    testset_new, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

for ix in range(args.run):
    print(f"\n===============> Experiment {ix}:")
    outfile = f'out/cifar10_maxiter{args.maxiter}_{args.i}_{ix}.pkl'
    experiment(all_trainloader, all_trainset, testloader, testloader_new, 
    outfile, maxiter = args.maxiter)

