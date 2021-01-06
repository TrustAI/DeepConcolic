import torch.nn as nn
import torch
import torch.nn.functional as F


class Basic_CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Basic_CNN, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1_1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(7*7*64, 200)
        self.fc2 = nn.Linear(200, self.num_classes)

    def forward(self, x):

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))

        x = self.maxpool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))

        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


        
class modelA(nn.Module):
    def __init__(self):
        super(modelA, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 20 * 20, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class modelB(nn.Module):
    def __init__(self):
        super(modelB, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, 64, 8)
        self.conv2 = nn.Conv2d(64, 128, 6)
        self.conv3 = nn.Conv2d(128, 128, 5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 12 * 12, 10)

    def forward(self, x):
        x = self.dropout1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout2(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x

class modelC(nn.Module):
    def __init__(self):
        super(modelC, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.conv2 = nn.Conv2d(128, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

