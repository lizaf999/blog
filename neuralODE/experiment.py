import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torchsummary import summary

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from adjoint import flat_parameters

from tqdm import tqdm

from NeuralODE import Model
#from ResNet18_torch import Model

#from pytorch tutorials
class Learner:
    def __init__(self,model,optimizer,criterion):
        self.device = torch.device('cuda:0')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, train_loader, epoch_num):
        self.model.train()
        
        for i in range(epoch_num):
            running_loss = 0.0

            with tqdm(train_loader,leave=True) as pbar:
                for batch_idx, data in enumerate(pbar):
                    self.optimizer.zero_grad()

                    inputs_cpu, labels_cpu = data

                    inputs = inputs_cpu.to(self.device)
                    labels = labels_cpu.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs,labels)
                    torch.autograd.set_detect_anomaly(True)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    if batch_idx % 10 == 9:
                        #print('[%d loss: %.3f' %(i + 1, running_loss / 10))
                        pbar.set_description('Progress %d/%d'%(i,epoch_num))
                        pbar.set_postfix(dict(loss=running_loss/10))
                        running_loss = 0.0
        
        print("Finished training}")


def experiment():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.31,), (0.13, ))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=False, num_workers=2)


    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    learner = Learner(model,optimizer,criterion)

    p = flat_parameters(model.parameters())
    print(p.size())
    summary(model,(1,28,28))

    #train and evaluate
    learner.train(trainloader,5)

    ##to save the learnt paramerters
    #PATH = "./trained"
    #torch.save(model.state_dict(), PATH)
    # model = Model()
    # model.load_state_dict(torch.load(PATH))
    # model = model.to(learner.device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images_cpu, labels_cpu = data
            images = images_cpu.to(learner.device)
            labels = labels_cpu.to(learner.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: %f %%' % (100 * correct / total))

if __name__ == "__main__":
    experiment()