import seaborn as sb
from matplotlib import pyplot as plt
import imageio
from PIL import Image
import h5py
import numpy as np
import os
import pandas as pd

label1 = pd.read_csv(os.path.join('result.csv'))
train = h5py.File(os.path.join('train', 'train_pre_data.h5'), 'r')
features = np.array(train['data'])
# 训练标签数据
labels = pd.read_csv(os.path.join('train', 'train_pre_label.csv'))


# 数据可视化plt  显示模型是否过拟合，如果loss和val loss 都一起下降的话未过拟合，一个下降一个上升就是过拟合
def plt_result(dataframe):
    fig = plt.figure(figsize=(16, 5))

    fig.add_subplot(1, 3, 1)
    plt.plot(dataframe['epoch'], dataframe['train_loss'], 'bo', label='Train loss')
    plt.plot(dataframe['epoch'], dataframe['val_loss'], 'b', label='Val loss')
    plt.title('Training and validation loss')
    plt.legend()

    fig.add_subplot(1, 3, 2)
    plt.plot(dataframe['epoch'], dataframe['train_acc'], 'bo', label='Train Accuracy')
    plt.plot(dataframe['epoch'], dataframe['val_acc'], 'b', label='Val Accuracy')
    plt.title('Training and validation Accuracy')
    plt.legend()

    fig.add_subplot(1, 3, 3)  # 画布的位置
    plt.plot(dataframe['epoch'], dataframe['train_f1_score'], 'bo', label='Train F1 Score')
    plt.plot(dataframe['epoch'], dataframe['val_f1_score'], 'b', label='Val F1 Score')
    plt.title('Training and validation F1 Score')
    plt.legend()

    plt.show()


# # 查看数据
# print(features.shape)
# print(labels.shape)
# print(labels.head())
#
# # 查看第一个数据的3D图像
# one_sample = features[0, 0]
# print(one_sample[0])
# # 读取图片,灰度化，并转为数组
# img = Image.fromarray(one_sample[40]).convert('L')
# # 放大四倍以便观察79*4,95*4
# img.resize((79, 95), Image.ANTIALIAS).save('image/hcgz.jpg')

# 查看训练集h5文件第300个数据第79层图像切片数据 #有79层切片
one_sample = features[299, 0]
print(one_sample[78])

# # 查看第300个训练数据的79层切片图像和动态图，图像之间间隔0.1秒
# one_sample = features[299, 0]
# frames = []
# for index in range(len(one_sample)):
#     img = Image.fromarray(one_sample[index]).convert('L')
#     img.resize((79 * 3, 95 * 3), Image.ANTIALIAS).save('image/temp{}.jpg'.format(index))
#     frames.append(imageio.imread('image/temp{}.jpg'.format(index)))
# imageio.mimsave('image/{0}.gif'.format('1'), frames, 'GIF', duration=0.1)  # 持续时间
#
# # 分析训练数据样本的组成
# order = labels['label'].value_counts()
# print(order)
# labels['label'].value_counts().plot.bar()
# plt.savefig("image/训练样本分布.png")
# plt.show()
#
# # 训练数据的分布
# order = label1['label'].value_counts().index
# sb.countplot(data=label1, x='label', order=order)
# plt.savefig("image/结果样本分布1.png")
# plt.show()
