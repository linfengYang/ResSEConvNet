import os
import argparse
import csv
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
from my_dataset import MyDataSet
from ResSEConvNet import model_architecture as get_model
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 64
    data_transform = {
        "train": transforms.Compose([transforms.Resize(img_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        "val": transforms.Compose([
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

    batch_size = args.batch_size
    nw = 6 # min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)
        filename1 = 'result/train_acc.csv'
        train_acc_row = train_acc
        with open(filename1, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([train_acc_row])
        # validate
        val_loss, val_acc,precision, recall,val_f1_test= evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        filename2 = 'result/val_acc.csv'
        val_acc_row = val_acc
        with open(filename2, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([val_acc_row])
        # 写入tensorboard,可在终端输入tensorboard --logdir=./  根据给出的http://localhost:6006/ 查看实时acc及各项指标
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # tb_writer.add_scalar(tags[5], val_f1, epoch)

        if best_acc < val_acc:
            n_count = 0
            torch.save(model.state_dict(), "./weights/best_model.pth")
            best_acc = val_acc
            print('best_acc:',best_acc)
            print('best_f1_:', val_f1_test)
            print('best_precision:',precision)
            print('best_recall:', recall)
            filename = '0720result/best_result.csv'
            acc_row = ['最佳acc', best_acc]
            val_f1_test_row = ['最佳f1', val_f1_test]
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([acc_row])
                writer.writerow([val_f1_test_row])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=11) 
    parser.add_argument('--epochs', type=int, default=130)  
    parser.add_argument('--batch-size', type=int, default=16)  
    parser.add_argument('--lr', type=float, default=8e-4)  
    parser.add_argument('--wd', type=float, default=5e-2)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default="the dataset path/")  

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
