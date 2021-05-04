import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd

class SimpleNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.fc_layers = nn.Sequential(
      nn.Linear(4, 5),
      nn.ReLU(),
      nn.Linear(5, 5),
      nn.ReLU(),
      nn.Linear(5, 5),
      nn.ReLU(),
      nn.Linear(5, 1)
    )
    self.loss_criterion = nn.L1Loss()


  def forward(self, x: torch.tensor) -> torch.tensor:
    model_output = self.fc_layers.forward(x)

    return model_output
    
class BankDataset(data.Dataset):
    def __init__(self, path):
        train = pd.read_csv(path,names=["variance","skewness","curtosis","entropy","label"])
        train_labels = train['label'].values
        train = train.drop("label",axis=1).values
        self.datalist = train
        self.labels = train_labels
    def __getitem__(self, index):
        return torch.Tensor(self.datalist[index].astype(float)), self.labels[index]
    def __len__(self):
        return self.datalist.shape[0]


class torchnn:

    def get_optimizer(self, model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:

        optimizer = None

        optimizer_type = config.get("optimizer_type", "sgd")
        learning_rate = config.get("lr", 1e-20)
        weight_decay = config.get("weight_decay", 1e-3)
        momentum = config.get("momentum",0.5)

        if optimizer_type == 'sgd':
            optimizer= torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        elif optimizer_type == 'adam':
            optimizer= torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            optimizer= torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'rms':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            pass

        return optimizer

    def __init__(self, optimizer_config=None, tanh=False):
        if optimizer_config == None:
            optimizer_config = {
                "optimizer_type": "adam",
                "lr": 1e-3,
                "weight_decay": 3e-3
            }
        self.Net = SimpleNet()
        if tanh:    
            self.Net.fc_layers = nn.Sequential(
                nn.Linear(4, 5),
                nn.Tanh(),
                nn.Linear(5, 5),
                nn.Tanh(),
                nn.Linear(5, 5),
                nn.Tanh(),
                nn.Linear(5, 1)
            )
        self.optimizer = self.get_optimizer(self.Net, optimizer_config)
    
    def train(self, trainpath, num_epoch):
        trainloader = BankDataset(trainpath)
        for epoch in range(num_epoch):  # loop over the dataset multiple times
        
            running_loss = 0.0
            for i in range(trainloader.datalist.shape[0]):
                # get the inputs; data is a list of [inputs, labels]
                labels = trainloader.labels[i]
                inputs = torch.Tensor(trainloader.datalist[i]).reshape(1,-1)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.Net(inputs)
                loss = (outputs - labels)**2
                loss.backward()
                self.optimizer.step()

                '''# print statistics
                running_loss += loss.item()
                if i % 100 == 0: 
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0'''

    def test(self, testpath):
        trainloader = BankDataset(testpath)
        #print(type(trainloader.datalist),trainloader.datalist.shape, trainloader.datalist[0])
        #print(self.Net(trainloader.datalist[0]))
        correct = 0
        for i in range(trainloader.datalist.shape[0]):
            output = self.Net(torch.Tensor(trainloader.datalist[i]))
            if (output[0].item() < 0.5) == (trainloader.labels[i] < 0.5):
                correct+=1
        print("Accuracy of model:",correct / trainloader.datalist.shape[0])


