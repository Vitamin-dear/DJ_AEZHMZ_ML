import torch
import os
import h5py
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import data_Als
import test
from DataSet import MyDataset
from sklearn.metrics import f1_score
import time
from Medicalnet import generate_model

# 79层79*95图像
input_D = 79
input_H = 95
input_W = 79

# 通道数
num_seg_classes = 3
# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 训练h5数据
train = h5py.File(os.path.join('train', 'train_pre_data.h5'), 'r')
features = np.array(train['data'])
# 训练标签数据
labels = pd.read_csv(os.path.join('train', 'train_pre_label.csv'))

# 预训练模型
pretrain_resnet_10_model = 'resnet_10.pth'
# 训练模型保存位置
medicanet_resnet10_model_path = 'medicanet_resnet10_model.pth'
medicanet_resnet10_result_ = 'result.csv'
# 损失函数
criterion = nn.CrossEntropyLoss()


def save_model(epochs, optimizer, model, filepath):
    checkpoint = {'epochs': epochs,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'model_state_dict': model.state_dict()}
    torch.save(checkpoint, filepath)


def load_model(filepath, phase='train', device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    model, _ = generate_model(sample_input_W=input_W,
                              sample_input_H=input_H,
                              sample_input_D=input_D,
                              num_seg_classes=num_seg_classes,
                              phase=phase,
                              pretrain_path=pretrain_resnet_10_model)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def train_data(model, train_dataloaders, valid_dataloaders, epochs, optimizer, scheduler, criterion, checkpoint_path,device='cpu'):
    # optimizer:优化器 scheduler:学习率动态调整函数 criterion:误差函数

    start = time.time()
    model_indicators = pd.DataFrame(
        columns=['epoch', 'train_loss', 'train_acc', 'train_f1_score', 'val_loss', 'val_acc', 'val_f1_score'])
    steps = 0
    n_epochs_stop = 10
    min_val_f1_score = 0
    epochs_no_improve = 0

    model.to(device)

    for e in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        train_correct_sum = 0
        train_simple_cnt = 0
        train_f1_score = 0
        y_train_true = []
        y_train_pred = []
        for ii, (images, labels) in enumerate(train_dataloaders):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct_sum += (labels.data == train_predicted).sum().item()
            train_simple_cnt += labels.size(0)
            y_train_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
            y_train_pred.extend(np.ravel(np.squeeze(train_predicted.cpu().detach().numpy())).tolist())

        scheduler.step()
        val_acc = 0
        val_correct_sum = 0
        val_simple_cnt = 0
        val_loss = 0
        val_f1_score = 0
        y_val_true = []
        y_val_pred = []
        with torch.no_grad():
            model.eval()
            for ii, (images, labels) in enumerate(valid_dataloaders):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, val_predicted = torch.max(outputs.data, 1)

                val_correct_sum += (labels.data == val_predicted).sum().item()
                val_simple_cnt += labels.size(0)

                y_val_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
                y_val_pred.extend(np.ravel(np.squeeze(val_predicted.cpu().detach().numpy())).tolist())

        train_loss = train_loss / len(train_dataloaders)
        val_loss = val_loss / len(valid_dataloaders)
        train_acc = train_correct_sum / train_simple_cnt
        val_acc = val_correct_sum / val_simple_cnt
        train_f1_score = f1_score(y_train_true, y_train_pred, average='macro')
        val_f1_score = f1_score(y_val_true, y_val_pred, average='macro')
        model_indicators.loc[model_indicators.shape[0]] = [e, train_loss, train_acc, train_f1_score, val_loss, val_acc,
                                                           val_f1_score]
        # 早期停止,根据模型训练过程中在验证集上的损失来保存表现最好的模型
        if val_f1_score > min_val_f1_score:
            save_model(e + 1, optimizer, model, checkpoint_path)
            epochs_no_improve = 0
            min_val_f1_score = val_f1_score
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')

    data_Als.plt_result(model_indicators)  # 可视化
    end = time.time()
    runing_time = end - start
    print('Training time is {:.0f}m {:.0f}s'.format(runing_time // 60, runing_time % 60))

X_train, X_val, y_train, y_val = train_test_split(features, labels['label'].values, test_size=0.2, random_state=42,stratify=labels['label'].values)

# 加载数据
train_datasets = MyDataset(datas=X_train, labels=y_train, shape=3, input_D=input_D, input_H=input_H, input_W=input_W,phase='train')
val_datasets = MyDataset(datas=X_val, labels=y_val, shape=3, input_D=input_D, input_H=input_H, input_W=input_W,phase='train')
train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=8, shuffle=False)

# 使用预训练模型的权重，迁移学习
medicanet_resnet3d_10, parameters = generate_model(sample_input_W=input_W,
                                                   sample_input_H=input_H,
                                                   sample_input_D=input_D,
                                                   num_seg_classes=num_seg_classes,
                                                   phase='train',
                                                   pretrain_path=pretrain_resnet_10_model)
params = [
    {'params': parameters['base_parameters'], 'lr': 0.001},
    {'params': parameters['new_parameters'], 'lr': 0.001 * 100}
]

# weight_decay=1e-3
# weight_decay=3e-4
# 优化器
optimizer = optim.Adam(params, weight_decay=3e-4)
# 学习率调整策略
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
# 学习层数
epochs = 100
train_data(medicanet_resnet3d_10, train_loader, val_loader, epochs, optimizer, scheduler, criterion,
           medicanet_resnet10_model_path, device)  # help.device

# 测试数据集加载
test_datasets = MyDataset(datas=test.test_data, shape=3, input_D=input_D, input_H=input_H, input_W=input_W,phase='test')
test_loader = DataLoader(dataset=test_datasets)

loadmodel = load_model(medicanet_resnet10_model_path, 'test', device)

test.all_predict(test_loader, loadmodel, device, medicanet_resnet10_result_)
