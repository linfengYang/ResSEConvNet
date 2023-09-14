# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
print(torch.cuda.is_available())
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer
# from models.ConvNext import convnext_tiny
from models.ConvNext_test import convnext_tiny

from torchvision import transforms
from my_dataset import MyDataSet
from utils_ConvNext import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') #  8e-4  resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
# parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='mlpmixer')  # swin vit_tiny vit_small cait_small convmixer ConvNext_test--------
parser.add_argument('--bs', default='16') # 32 for convmixer;
parser.add_argument('--size', default="64") # 32-------
parser.add_argument('--n_epochs', type=int, default='200') # 200
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

args = parser.parse_args()

# take in args
# usewandb = ~args.nowandb
# if usewandb:
#     import wandb
#     watermark = "{}_lr{}".format(args.net, args.lr)
#     wandb.init(project="cifar10-challange",
#             name=watermark)
#     wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = bool(~args.noamp)
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize
# Prepare dataset
# --------old preprocessing---------
# transform_train = transforms.Compose([
#     transforms.RandomCrop(64, padding=4), # 32
#     transforms.Resize(size),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.Resize(size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# # Add RandAugment with N, M(hyperparameter)
# if aug:  
#     N = 2; M = 14;
#     transform_train.transforms.insert(0, RandAugment(N, M))

# from torchvision import datasets, transforms
# from sklearn.model_selection import train_test_split
# # transform = transforms.Compose([
# #     transforms.Resize((64, 64)),
# #     transforms.ToTensor()
# # ])
# data_path ='/content/drive/MyDrive/Vit/vision-transformers-cifar10-main/vision-transformers-cifar10-main/data/app_GAF/'
# trainset = datasets.ImageFolder(data_path+'train', transform=transform_train)
# testset = datasets.ImageFolder(data_path+'val', transform=transform_test)
# # train_dataset = torch.utils.data.TensorDataset(x, y)  # 对给定的 tensor 数据，将他们包装成 dataset
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=1)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)
# --------old preprocessing---------

# !!!-----new preprocessing----------
data_path ='../vmdwpt_db6_L3_imif_DJX_0913/'  # vmdwpt_db6_L3_imif_0817  GAF   MTF  MDF  RP  
train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)

img_size = 64  # 224   # ------------
data_transform = {
    "train": transforms.Compose([#transforms.RandomResizedCrop(img_size),  # ---------
                  transforms.Resize(img_size),
                  # transforms.RandomHorizontalFlip(), # ---------
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # GAF每个像素值在[-1,1]之间，不再进行归一化
                  ]),
    "val": transforms.Compose([# transforms.Resize(int(img_size * 1.143)),
                  #transforms.CenterCrop(img_size),  # 没必要吧？
                  transforms.Resize(img_size),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])}

# 实例化训练数据集
train_dataset = MyDataSet(images_path=train_images_path,
              images_class=train_images_label,
              transform=data_transform["train"])

# 实例化验证数据集
val_dataset = MyDataSet(images_path=val_images_path,
            images_class=val_images_label,
            transform=data_transform["val"])
batch_size = bs
nw = 4 # min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))
trainloader = torch.utils.data.DataLoader(train_dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      pin_memory=True,
                      num_workers=nw,
                      collate_fn=train_dataset.collate_fn)

testloader = torch.utils.data.DataLoader(val_dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      pin_memory=True,
                      num_workers=nw,
                      collate_fn=val_dataset.collate_fn)
# !!!---------------


# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    print('------------------------welcome convmixer!------------------------')
    net = ConvMixer(256, 16, kernel_size=4, patch_size=1, n_classes=11) # kernel_size=args.convkernel
elif args.net=="swin":
    print('------------------------welcome swin!------------------------')
    from models.swin import swin_t
    net = swin_t(window_size= 8,num_classes=11,downscaling_factors=(2,2,2,1))  # window_size=args.patch
elif args.net == "ConvNext_test":
    print('------------------------welcome ConvNext_test!------------------------')
    net = convnext_tiny(num_classes=11)
elif args.net == "ConvNext":
    print('------------------------welcome ConvNext!------------------------')
    net = convnext_tiny(num_classes=11)
elif args.net=="mlpmixer":
    print('------------------------welcome mlpmixer!------------------------')
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = size,
    channels = 3,
    patch_size =  8, # args.patch,
    dim = 512,
    depth = 6,
    num_classes = 11  # ---------
)
elif args.net=="vit_small":
    from models.vit_small import ViT
    print('------------------------welcome vit_small!------------------------')
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 11, # ---------
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    print('------------------------welcome vit_tiny!------------------------')
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 11, # --
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    print('------------------------welcome simplevit!------------------------')
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 11,  # ------------
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
)
elif args.net=="vit":
    # ViT for cifar10
    print('------------------------welcome vit!------------------------')
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 11,  # --------
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="cait":
    from models.cait import CaiT
    print('------------------------welcome cait!------------------------')
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 11,  # ---------
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="cait_small":
    print('------------------------welcome cait_small!------------------------')
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 11, # -----------
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)

# For Multi-GPU
if 'cuda' in device:
    print(device)
    print("using data parallel")
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
       
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

# 计算f1 ---------------
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, precision_score, \
    recall_score, f1_score, confusion_matrix, accuracy_score
# 计算f1 ---------------

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    labels_list = torch.empty(0)
    predicts_list = torch.empty(0)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets_f1 = targets
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # 计算f1 ---------------
            targets_cpu = targets.cpu()
            predicted_cpu = predicted.cpu()
            labels_list = torch.cat([labels_list, targets_cpu], dim=0)
            predicts_list = torch.cat([predicts_list, predicted_cpu], dim=0)
            # 计算f1 ---------------
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    labels_list = labels_list.to(torch.int32)
    predicts_list = predicts_list.to(torch.int32)
    f1_test = f1_score(y_true=labels_list, y_pred=predicts_list, average='macro')
    print('f1:',f1_test)
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f},F1:{(f1_test):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

# if usewandb:
#     wandb.watch(net)
    
net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    # if usewandb:
    #     wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
    #     "epoch_time": time.time()-start})

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)

# writeout wandb
# if usewandb:
#     wandb.save("wandb_{}.h5".format(args.net))
    
