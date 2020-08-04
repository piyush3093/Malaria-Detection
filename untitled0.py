import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

transform = transforms.Compose([transforms.RandomResizedCrop(200),
                                transforms.ToTensor()])
data = datasets.ImageFolder('cell_images/', transform = transform)

import matplotlib.pyplot as plt
classes = ['P', 'U']
len(data)

batch_size = 20
num_workers = 0
indices = [i for i in range(len(data))]
np.random.shuffle(indices)
split = int(np.floor(0.75*len(data)))
train_idx, valid_idx = indices[:split], indices[split:]
train_idx = train_idx[:20660]
valid_idx = valid_idx[:6880]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, num_workers = num_workers, sampler = train_sampler)
valid_loader = torch.utils.data.DataLoader(data, batch_size = batch_size, num_workers = num_workers, sampler = valid_sampler)

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

len(dataiter)

print(labels)

images[1].shape

fig = plt.figure(figsize = (25, 4))
for idx in range(20):
    ax = fig.add_subplot(2, 20/2, idx + 1, xticks = [], yticks = [])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*25*25, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128*25*25)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = Net()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
n_epochs = 25

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0
    
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    
    model.eval()
    for data, target in valid_loader:
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        
    train_loss += train_loss/len(train_loader.sampler)
    valid_loss += valid_loss/len(valid_loader.sampler)
    
    print("Epoch {}, Training Loss {:.6f}, Valid Loss {:.6f}".format(epoch, train_loss, valid_loss))
    








