import pygame as py
import _thread
import time
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import multiprocessing
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from freetype import *
from PIL import ImageFont,ImageDraw,Image
import keras.backend as K
from keras.layers import Flatten, Dense, BatchNormalization
from keras.models import Model
from keras import Input, regularizers
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as Data
import math
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Resize,Grayscale,RandomHorizontalFlip
import torch.utils.model_zoo as model_zoo
import gc
import torchvision.datasets as dset
gc.collect()
import tensorflow as tf
from tensorflow.python.training import moving_averages
from torch.nn import functional as F
import matplotlib.pyplot as plt
from freetype import *
from PIL import ImageFont,ImageDraw,Image

window_width = 960
window_height = 720
image_width = 720
image_height = 480
imagepos_x = 0
imagepos_y = 0
butpos_x = 450
butpos_y = 450

model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    }


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=26):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def resnet18(pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def input_transform():
    return Compose([
        # Grayscale(num_output_channels=1),
        Resize((224, 224)),  # 改变尺寸
        ToTensor(),  # 变成tensor
        RandomHorizontalFlip(),
    ])




# 图像转换，用于在画布中显示
def tkImage(vc):
    ref, frame = vc.read()
    cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(cvimage)
    pilImage = pilImage.resize((image_width, image_height), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=pilImage)
    return tkImage

# 图像的显示与更新
def video():
    def video_loop():
        try:
            while True:
                picture1 = tkImage(vc1)
                canvas1.create_image(0, 0, anchor='nw', image=picture1)
                win.update_idletasks()  # 最重要的更新是靠这两句来实现
                win.update()
        except:
            pass

    video_loop()
    win.mainloop()
    vc1.release()
    cv2.destroyAllWindows()

def hit():
    global vc1,e

    videofile_path = e.get()
    videos = os.listdir(videofile_path)
    for video_name in videos:
        file_name = video_name.split('.')[0]  # 视频文件名
        # print(file_name)
        folder_name = videofile_path + '/' + file_name
        # print(folder_name)
        os.makedirs(folder_name, exist_ok=True)  # 创建与视频同名目录保存截取的图片
        # print(videofile_path + video_name)
    # print(videofile_path + video_name)

    cap = cv2.VideoCapture(videofile_path + '/' + video_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取视频每秒的帧数
    print("fps: ", fps)
    num = 0
    cnt = 0  # 帧计数
    time = 200  # 保存帧间隔
    if cap.isOpened():
        while True:
            ret, img = cap.read()  # img 就是一帧图片
            # print(ret,img)
            pic_path = folder_name + '//'
            # print(folder_name)
            if ret and cnt % time == 0:
                num = num + 1
                # print(1)
                pic_name = str('%04d' % num) + '.jpg'
                # print(pic_name)
                # print(pic_path + pic_name)  # 打印生成的路径名
                cv2.imencode('.jpg', img)[1].tofile(pic_path + pic_name)
                cv2.waitKey(1)
            # cv2.imshow('pic1',img)
            # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
            if cnt > 5000:
                break
            cnt = cnt + 1
        cap.release()
        cv2.destroyAllWindows()
    else:
        print('视频打开失败！')





    BATCH_SIZE = 10

    test_data = dset.ImageFolder(root=videofile_path, transform=input_transform())

    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=0)  # 测试集

    net = torch.load('D:/newdir\sjj\American Sign Language Letters.v1-v1.yolov5pytorch\model/model8.pth',
                     map_location=torch.device('cpu'))

    result = []
    for batch in test_loader:  # GetBatch
        images, labels = batch
        outs = net(images)  # PassBatch
        _, pre = torch.max(outs.data, 1)
        result += pre.numpy().tolist()

    # picfile_path=videofile_path+file_name
    picfile_path = videofile_path + '/' + file_name
    pics = os.listdir(picfile_path)
    # print(pics)
    word = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
    for pic_name in pics:
        file_name1 = pic_name.split('.')[1]  # 测试完成后的文件名
        # print(file_name1)
        folder_name1 = picfile_path + file_name1
        # print(folder_name1)
        os.makedirs(folder_name1, exist_ok=True)  # 创建与测试完成后的文件夹同名目录保存截取的图片

    # print(picfile_path+file_name1)


    cnt = 1
    for i in range(len(pics)):
        picname = picfile_path + '/' + pics[i]
        print(i)
        image = Image.open(picname)
        # print(image)
        font = ImageFont.truetype('consola.ttf', 90)  # 设置字体及字号
        drawobj = ImageDraw.Draw(image)
        text = word[result[i]]  # 列表索引即可
        # 位置 文本 颜色
        drawobj.text([0, 0], text, 'red', font)
        # image.show()
        image.save(picfile_path + file_name1 + '/' + pics[i])  # 保存路径
        cnt = cnt + 1


    os.system(f'ffmpeg -loop 1 -f image2 -i ' + folder_name1 + '/%04d.jpg -vcodec libx264 -r 10 -t 30 D:/newdir/shouyu/test1.mp4')

    vc1=cv2.VideoCapture(videofile_path+'/test1.mp4')
    video()

'''布局'''
win = tk.Tk()
win.geometry('1000x700')
canvas1 = Canvas(win, bg='white', width=image_width, height=image_height)
canvas1.pack(side='bottom')

e=tk.Entry(win)
e.pack(side='top')

b=tk.Button(win,text='开始',width=15,height=2,command=hit).pack(side='top')

win.mainloop()
