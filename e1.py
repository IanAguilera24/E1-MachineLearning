import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class ConvNet(nn.Module):

    def __init__(self):

        super(ConvNet, self).__init__()

        self.cn1 = nn.Conv2d(1, 16, 3, 1)

        self.cn2 = nn.Conv2d(16, 32, 3, 1)

        self.dp1 = nn.Dropout2d(0.10)

        self.dp2 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(4608, 64) # 4608 is basically 12 X 12 X 32

        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):

        x = self.cn1(x)

        x = F.relu(x)

        x = self.cn2(x)

        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = self.dp1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = F.relu(x)

        x = self.dp2(x)

        x = self.fc2(x)

        op = F.log_softmax(x, dim=1)

        return op
    
def train(model, device, train_dataloader, optim, epoch):

    model.train()

    for b_i, (X, y) in enumerate(train_dataloader):

        X, y = X.to(device), y.to(device)

        optim.zero_grad()

        pred_prob = model(X)

        loss = F.nll_loss(pred_prob, y) # nll is the negative likelihood loss

        loss.backward()

        optim.step()

        if b_i % 10 == 0:

            print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(

                epoch, b_i * len(X), len(train_ dataloader.dataset),

                100. * b_i / len(train_dataloader), loss. item()))