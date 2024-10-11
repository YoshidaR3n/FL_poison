from collections import OrderedDict
from typing import List, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower."
)
disable_progress_bar()


args = sys.argv
ID = args[1]
############################################################
#クロスサイロ化するためにデータを分割
############################################################

NUM_CLIENTS = 3
BATCH_SIZE = 32
POISON = True

def load_datasets():
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})

    def apply_transforms(batch):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []

    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8)

        if(partition_id==2 and POISON==True):
            for data in partition["train"]:
                if data["label"] == 3:
                    data["label"] = 4
        
        trainloaders.append(DataLoader(partition["train"], batch_size=BATCH_SIZE))
        valloaders.append(DataLoader(partition["test"], batch_size=BATCH_SIZE))

    testset = fds.load_full("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


trainloaders, valloaders, testloader = load_datasets()



############################################################
#モデルの定義・学習(連合)
############################################################

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        """
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        """
        self.conv1=nn.Conv2d(3,64,3) #32-3=29
        self.conv2=nn.Conv2d(64,128,3) #29-3=26
        self.pool1=nn.MaxPool2d(2,2) #26/2=13
        self.conv3=nn.Conv2d(128,256,3) #13-3=10
        self.conv4=nn.Conv2d(256,512,2) #10-2=8
        self.pool2=nn.MaxPool2d(2,2) #8/2=4
        self.fc1=nn.Linear(512 * 4 * 4, 1024)
        self.fc2=nn.Linear(1024, 256)
        self.fc3=nn.Linear(256, 10)
        self.relu=F.relu



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
def train(net, trainloader, epochs: int, verbose=True):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    print(f"一つのデータセットの大きさ{len(trainloaders[0])}")

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

trainloader = trainloaders[int(ID)]
valloader = valloaders[int(ID)]
net = Net().to(DEVICE)

############################################################
#連合学習
############################################################
def set_parameters(net, parameters: List[np.ndarray]):
    #print("receve:")
    #print(parameters[0][0][0][0])
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    #print("send:")
    #a=[val.cpu().numpy() for _, val in net.state_dict().items()]
    #print(a[0][0][0][0])
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class FlowerClient(fl.client.NumPyClient):

    """
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
    """

    def get_parameters(self, config):
        return get_parameters(net)

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        train(net, trainloader, epochs=5)
        loss, accuracy = test(net, valloader)
        print(f"ClientID:{ID} validation loss {round(loss,4)}, accuracy {round(accuracy,4)}")
        return get_parameters(net), len(trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, valloader)
        return float(loss), len(valloader), {"accuracy": float(accuracy)}
    
############################################################
#接続
############################################################
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())



    