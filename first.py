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
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from freetype import *
from PIL import ImageFont,ImageDraw,Image






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
                #Grayscale(num_output_channels=1),
                Resize((224,224)),   #改变尺寸
                ToTensor(),      #变成tensor
                RandomHorizontalFlip(),
                ])



BATCH_SIZE = 10

data=dset.ImageFolder(root=r"D:\newdir\sjj\American Sign Language Letters.v1-v1.yolov5pytorch\train1",transform=input_transform())
test_data=dset.ImageFolder(root=r"D:\newdir\sjj\American Sign Language Letters.v1-v1.yolov5pytorch\test1",transform=input_transform())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


loader=torch.utils.data.DataLoader(dataset=data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)#训练集

# train_loader=torch.utils.data.DataLoader(dataset=train,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)#训练集
test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)#测试集











net  = resnet18().to(device)

#net=torch.load('D:/newdir\sjj\American Sign Language Letters.v1-v1.yolov5pytorch\model/model8.pth',map_location=torch.device('cpu'))

optimizer=optim.SGD(net.parameters(),lr=0.1,weight_decay=0.01)



#
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss().to(device)
for epoch in range(50):
    correct = 0
    total = 0
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x).to(device)
        b_y = Variable(batch_y).to(device)
        predict = net(b_x).to(device)
        loss = loss_func(predict, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(predict.data, 1)
        total += batch_y.size(0)
        correct += (predicted == b_y).sum()

        if step % 50 == 0:
            print('epoch:{}, step:{}, loss:{}, acc:{}'.format(epoch, step, loss, correct/total))

torch.save(net,'D:/newdir\sjj\American Sign Language Letters.v1-v1.yolov5pytorch\model/model8.pth')

# correct = 0
# total = 0
# for step, (batch_x, batch_y) in enumerate(test_loader):
#
#     #batch_x = Variable(torch.unsqueeze(batch_x, dim=0).float(), requires_grad=False)
#     #print(batch_x.shape)
#     b_x = Variable(batch_x)
#     b_y = Variable(batch_y)
#
#
#
#     predict = net(b_x)
#
#     _, predicted = torch.max(predict.data, 1)
#     total += batch_y.size(0)
#     #total += 1
#     correct += (predicted == b_y).sum()
#     result += predicted.numpy().tolist()
#print(' step:{},  acc:{}'.format(step, correct / total))
result=[]
for batch in test_loader:  # GetBatch
    images, labels = batch
    outs = net(images)  # PassBatch
    _, pre = torch.max(outs.data, 1)
    result += pre.numpy().tolist()


print(len(result))
