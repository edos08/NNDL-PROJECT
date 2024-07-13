import time
import torch

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm


class ConvolutionalModel(nn.Module):
    """
    Simple Convolutional Neural Network for image classification using four convolutional layers
    """

    def __init__(self, output_size, dropout=0.32):
        super().__init__()

        self.net = nn.Sequential(
            # ConvBlock 1
            nn.Conv2d(3, 6, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5, padding=0),

            # ConvBlock 2
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # ConvBlock 3
            nn.Conv2d(16, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0),

            # ConvBlock 4
            nn.Conv2d(32, 120, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            nn.Flatten(),

            nn.Dropout(p=dropout),

            # DenseBlock
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_size)
        )

    def forward(self, x):
        output = self.net(x)
        return output

# Resnet implementation inspired by https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

# Basic residual block, used in shallower Resnets (resnet18 and resnet34)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # ConvBlock 1
        self.conv1 = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # ConvBlock 2
        self.conv2 = nn.Sequential(
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = out_channels  # not actually used
        self.stride = stride  # not actually used

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out
    
# Residual block using Bottleneck architecture, used in deeper Resnets (from 50 layers upwards)
class BottleneckBlock(nn.Module): # NOTE: stride applied at the 3x3 conv layer
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Sequential(
            conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential( # NOTE: stride is applied here
            conv3x3(out_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            conv1x1(out_channels, self.expansion * out_channels),
            nn.BatchNorm2d(self.expansion * out_channels)
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.out_channels = self.expansion * out_channels  # not actually used
        self.stride = stride  # not actually used

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Resnet implementation inspired by https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        
        self.inplanes = 64  # initial number of channel/feature expansion
        
        self.conv1 = nn.Sequential(  # initial downsampling blocks
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # actual residual layers
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        
        # dense classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # param initialization (same as in torchvision)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes*block.expansion: # downsample when dimension of residual and out don't match
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) # downsampling done in first block
        
        self.inplanes = planes * block.expansion # update inplanes var
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        # preprocess
        x = self.conv1(x)
        x = self.maxpool(x)

        # residual layers
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # classfication via dense layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
# from https://github.com/facebookarchive/fb.resnet.torch/blob/master/models/resnet.lua
resnet_cfg = {
    "resnet18" : {"block" : BasicBlock, "layers" : [2, 2, 2, 2]},
    "resnet34" : {"block" : BasicBlock, "layers" : [3, 4, 6, 3]},
    "resnet50" : {"block" : BottleneckBlock, "layers" : [3, 4, 6, 3]},
    "resnet101" : {"block" : BottleneckBlock, "layers" : [3, 4, 23, 3]},
    "resnet152" : {"block" : BottleneckBlock, "layers" : [3, 8, 36, 3]},
}

# example init: 
#   resnet50 = ResNet(resnet_cfg["ResNet50"]["block"], resnet_cfg["ResNet50"]["layers"], num_classes=10)


def train(train_loader: DataLoader, model: nn.Module, epoch: int, criterion: nn.modules.loss, optimizer: torch.optim,
          device, pbar: tqdm = None) -> (list, list):
    """
    Train a model on the provided training dataset

    Args:
         train_loader (DataLoader): training dataset
         model (nn.Module): model to train
         epoch (int): the current epoch
         criterion (nn.modules.loss): loss function
         optimizer (torch.optim): optimizer for the model
         device (torch.device): the device to load the model
         pbar (tqdm): tqdm progress bar
    """

    model.train()
    start = time.time()

    epoch_loss = list()
    pred_list, label_list = np.array([]), np.array([])

    for batch in train_loader:
        image, label = batch

        # GPU casting
        image = image.to(device)
        label = label.to(device)

        # Forward step
        pred_label = model(image)
        loss = criterion(pred_label, label)
        epoch_loss.append(loss.cpu().data)

        _, predicted = torch.max(pred_label, dim=1)
        pred_list = np.append(pred_list, predicted.cpu().numpy())
        label_list = np.append(label_list, label.cpu().numpy())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if pbar is not None:
            pbar.update(1)

    epoch_loss = np.asarray(epoch_loss)
    epoch_acc = accuracy_score(label_list, pred_list)
    end = time.time()

    print("\n-- TRAINING --")
    print("Epoch: {}\n - Loss: {:.3f} +- {:.3f}\n - Accuracy: {:.2f}\n - Time: {:.2f}s".format(epoch + 1,
                                                                                               epoch_loss.mean(),
                                                                                               epoch_loss.std(),
                                                                                               epoch_acc,
                                                                                               end - start))

    return epoch_loss.mean(), epoch_acc


def validate(validation_loader: DataLoader, model: nn.Module, epoch: int, criterion: nn.modules.loss, device,
             pbar: tqdm = None) -> (float, float):
    """
    Valid a model on the provided validation dataset

    Args:
        validation_loader (DataLoader): validation dataset
        model (nn.Module): the model to evaluate
        epoch (int): the current epoch
        criterion (torch.nn.modules.loss): loss function
        device (torch.device): the device to load the model
        pbar (tqdm): tqdm progress bar
    """

    model.eval()
    start = time.time()

    epoch_loss = list()
    pred_list, label_list = np.array([]), np.array([])

    with torch.no_grad():
        for batch in validation_loader:
            image, label = batch

            # Casting to GPU
            image = image.to(device)
            label = label.to(device)

            # Forward step
            pred_label = model(image)
            loss = criterion(pred_label, label)
            epoch_loss.append(loss.cpu().data)

            _, predicted = torch.max(pred_label, dim=1)
            pred_list = np.append(pred_list, predicted.cpu().numpy())
            label_list = np.append(label_list, label.cpu().numpy())

            if pbar is not None:
                pbar.update(1)

        epoch_loss = np.asarray(epoch_loss)
        epoch_acc = accuracy_score(label_list, pred_list)
        end = time.time()

        print("\n-- VALIDATION --")
        print("Epoch: {}\n - Loss: {:.3f} +- {:.3f}\n - Accuracy: {:.2f}\n - Time: {:.2f}s".format(epoch + 1,
                                                                                                   epoch_loss.mean(),
                                                                                                   epoch_loss.std(),
                                                                                                   epoch_acc,
                                                                                                   end - start))

        return epoch_loss.mean(), epoch_acc


def test(test_loader: DataLoader, model: nn.Module, criterion: nn.modules.loss, device) -> (list, list):
    """
    Test a model on the provided test dataset

    Args:
        test_loader (DataLoader): validation dataset
        model (nn.Module): the model to evaluate
        criterion (nn.modules.loss): loss function
    """

    model.eval()
    start = time.time()

    losses = list()
    pred_list, label_list = np.array([]), np.array([])

    with torch.no_grad():
        for batch in test_loader:
            image, label = batch

            # Casting to GPU
            image = image.to(device)
            label = label.to(device)

            # Forward step
            pred_label = model(image)
            loss = criterion(pred_label, label)
            losses.append(loss.cpu().data)

            _, predicted = torch.max(pred_label, dim=1)
            pred_list = np.append(pred_list, predicted.cpu().numpy())
            label_list = np.append(label_list, label.cpu().numpy())

        losses = np.asarray(losses)
        accuracy = accuracy_score(label_list, pred_list)
        end = time.time()

        print("\n-- TEST --")
        print("Test results:\n - Loss: {:.3f} +- {:.3f}\n - Accuracy: {:.2f}\n - Time: {:.2f}s".format(losses.mean(),
                                                                                                       losses.std(),
                                                                                                       accuracy,
                                                                                                       end - start))

        return losses.mean(), accuracy
