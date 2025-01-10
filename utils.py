import os
import sys
import json
import pickle
import random
import math
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):  # val_rate: float = 0.2---------
    random.seed(0)  
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  
    train_images_label = [] 
    val_images_path = []  
    val_images_label = [] 
    every_class_num = [] 
    supported = [".jpg", ".JPG", ".png", ".PNG"]  
   
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
     
        images.sort()
    
        image_class = class_indices[cla]
        
        every_class_num.append(len(images))
      
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
      
        plt.bar(range(len(flower_class)), every_class_num, align='center')
      
        plt.xticks(range(len(flower_class)), flower_class)
       
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
       
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([]) 
            plt.yticks([])  
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device) 
    accu_num = torch.zeros(1).to(device)   
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def train_one_epoch_K(model, optimizer, data_loader, device, epoch, lr_scheduler):

    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)   
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        images = images.float()
        pred = model(images.to(device))   #
        pred_classes = torch.max(pred, dim=1)[1]  
        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        labels_true = torch.argmax(labels, dim=1)  
        accu_num += torch.eq(pred_classes.to(device), labels_true.to(device)).sum()
        loss = loss_function(pred, labels.to(device).float())
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, precision_score, \
    recall_score, f1_score, confusion_matrix, accuracy_score

@torch.no_grad()
def test(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    y_true = 0
    y_pred = 0
    acc_mean=[]
    precision_mean = []
    recall_mean = []
    f1_mean = []
    model.eval()
    accu_num = torch.zeros(1).to(device)  
    accu_loss = torch.zeros(1).to(device)  
    sample_num = 0
    n = 0 
    data_loader = tqdm(data_loader, file=sys.stdout)
    for batch_idx, data in enumerate(data_loader):
        X, Y = data[0], data[1]
        X = X.to(device)
        Y = Y.to(device)
        sample_num += X.shape[0]
        # print('----sample_num:----',sample_num)
        # X_test, Y_test = X_test.to(device), Y_test.to(device)
        outputs = model(X)
        _, predicted = torch.max(outputs.data, dim=1)

        total += Y.size(0)
        correct += (predicted == Y).sum().item()
        # print('accuracy on test set: %.2f %% ' % (100 * correct / total))
        acc_mean.append(accuracy_score(Y.cpu(), predicted.cpu()))
        precision_mean.append(precision_score(Y.cpu(), predicted.cpu(),average='macro')) # macro   weighted
        recall_mean.append(recall_score(Y.cpu(), predicted.cpu(),average='macro'))
        f1_mean.append(f1_score(Y.cpu(), predicted.cpu(),average='macro'))
        accu_num += torch.eq(predicted, Y.to(device)).sum()
        loss = loss_function(outputs, Y.to(device))  # 
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (batch_idx + 1),
            accu_num.item() / sample_num
    )
    return accu_loss.item() / (batch_idx + 1), accu_num.item() / sample_num,np.mean(precision_mean),np.mean(recall_mean),np.mean(f1_mean)
@torch.no_grad()
def evaluate_K(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   
    accu_loss = torch.zeros(1).to(device) 
    f1 = torch.zeros(1).to(device)
    sample_num = 0
    n = 0
    labels_list = torch.empty(0)
    predicts_list = torch.empty(0)
   
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]  
        pred = model(images.to(device))  
        pred_classes = torch.max(pred, dim=1)[1]  

        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()  
        labels_true = torch.argmax(labels, dim=1)  
        accu_num += torch.eq(pred_classes.to(device), labels_true.to(device)).sum()
        loss = loss_function(pred, labels.to(device).float()) 
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

        labels_list = torch.cat([labels_list.to(device), labels.to(device)], dim=0)
        predicts_list = torch.cat([predicts_list.to(device), pred_classes.to(device)], dim=0)

    labels_list = labels_list.to(torch.int32)
    labels_list = torch.argmax(labels_list,dim=1)
    predicts_list = predicts_list.to(torch.int32)
    # print(labels_list)
    # print(predicts_list)
    labels_list = labels_list.cpu()
    predicts_list = predicts_list.cpu()
    f1_test = f1_score(y_true=labels_list, y_pred=predicts_list, average='macro')
    precision = precision_score(labels_list, predicts_list, average='macro')  # ---
    recall = recall_score(labels_list, predicts_list, average='macro')  # ---

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,precision, recall,f1_test
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  
    accu_loss = torch.zeros(1).to(device)
    f1 = torch.zeros(1).to(device)
    sample_num = 0
    n = 0 
    labels_list = torch.empty(0)
    predicts_list = torch.empty(0)
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]  
        pred = model(images.to(device))  
        pred_classes = torch.max(pred, dim=1)[1] 

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()  
        loss = loss_function(pred, labels.to(device))  
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )
        # 计算f1 ---------------
        labels_cpu = labels.cpu()
        pred_classes_cpu = pred_classes.cpu()

        labels_list = torch.cat([labels_list, labels_cpu], dim=0)
        predicts_list = torch.cat([predicts_list, pred_classes_cpu], dim=0)

    labels_list = labels_list.to(torch.int32)
    predicts_list = predicts_list.to(torch.int32)
    # print(labels_list)
    # print(predicts_list)
    f1_test = f1_score(y_true=labels_list, y_pred=predicts_list, average='macro')
    precision = precision_score(y_true=labels_list, y_pred=predicts_list, average='macro')  # ---
    recall = recall_score(y_true=labels_list, y_pred=predicts_list, average='macro')  # ---

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,precision, recall,f1_test
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):

    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
