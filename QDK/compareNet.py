import pandas
import torch
import json
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from prettytable import PrettyTable
from datetime import datetime
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm  # For Loading Bars

batch_size = 15
num_epochs = 10

# TODO: make sure to use the correct path
dataJson = json.load(open('../archives/HalfMoon/data_1000_0.3.json', 'r'))
train_features = pandas.DataFrame(dataJson['TrainingData']['Features'])
train_labels = pandas.DataFrame(dataJson['TrainingData']['Labels'])
val_features = pandas.DataFrame(dataJson['ValidationData']['Features'])
val_labels = pandas.DataFrame(dataJson['ValidationData']['Labels'])

# Remove second dimension from labels
train_labels = np.squeeze(train_labels)
val_labels = np.squeeze(val_labels)

# Create Tensor Dataset
train_set = TensorDataset(torch.tensor(train_features.values), torch.tensor(train_labels))
val_set = TensorDataset(torch.tensor(val_features.values), torch.tensor(val_labels))

# Create DataLoader for train and val(test) data
trainLoader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valLoader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

# ######################################################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        #self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(x)
        x = self.fc1(x)

        #x = F.relu(x)
        #x = self.fc2(x)

        return x


# Initialize network
my_net = net()
my_net.to(device)
print(my_net)

# Method to print the weights of the layer
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
count_parameters(my_net)

# loss and optimizer
criterion = nn.CrossEntropyLoss()  # already uses softmax so not needed in forward
optimizer = optim.Adam(my_net.parameters(), lr=0.01)

# Take single batch
# inputsS, labelsS = next(iter(trainLoader))  # inputs, labels = inputsS, labelsS (in train)

def train(model, train_loader):
    for epoch in range(num_epochs):  # no. of epochs
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_loss = 0.0
        model.train()
        # for batch_index, data in enumerate(train_loader, 0):
        for batch_index, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # get the inputs; data is a list of [inputs, labels]  # print(f"Feature batch shape: {inputs.size()}")
            inputs, labels = data
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            # forward
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            # print(loss)  # (only use if testing a single batch)

            # backward
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            train_loss = loss.item() * inputs.size(0)

        print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_loader)}')

    print('Done Training')


def check_accuracy(model, test_loader):
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(inputs.float())
            _, predictions = outputs.max(1)
            num_correct += (predictions == labels).sum().item()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} "
            f"with accuracy {float(num_correct) / float(num_samples) * 100:.2f} %"
        )


startTrainTime=datetime.now()
train(my_net, trainLoader)
print("Train Runtime: ", datetime.now() - startTrainTime)

check_accuracy(my_net, valLoader)
