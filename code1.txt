# data visualization and manipulation libraries
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Pytorch
import torch
import torchvision
import torch.nn.functional as F # Contains all the activation functions
import torch.nn as nn # The Neural Network
import torch.optim as optim # The Optimizer
import torchvision.transforms as transforms # The Tranform to be used

from torch.utils.data import DataLoader # Dataloader

##### 

# we first download the dataset without using any transform
training_data = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True)

# observe the shape of the data
print(training_data.data.shape)

####

# We just want to convert it to a tensor
training_transform = transforms.Compose([
    transforms.ToTensor()
])

# We use the same transform for the training and validation data
testing_transform = training_transform

# we download the data set again
# but
# this time we will be applying the transforms as well

training_data = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=training_transform)
testing_data = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=testing_transform)

# Making the dataloader
training_loader = DataLoader(training_data, batch_size = 128, shuffle = True, num_workers = 2)
testing_loader = DataLoader(testing_data, batch_size = 256, shuffle = False, num_workers = 2)

#####

# Defining the Classes present
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# defining the number of rows and cols to be displayed
rows = 3
cols = 3

# getting the samples from the training set
image = training_data.data[:rows*cols]
label = training_data.targets[:rows*cols]

fig = plt.figure(figsize=(2 * cols, 2 * rows))

i = 0

# Add subplot for each image
for col in range(cols):
    for row in range(rows):

        ax = fig.add_subplot(rows, cols, col * rows + row + 1) # Add a sub-plot at (row, col)
        # ax.grid(b=False) # Get rid of the grids
        ax.axis("off") # Get rid of the axis
        ax.imshow(image[i, :]) # Show random image
        ax.set_title(CIFAR10_CLASSES[label[i]]) # Set title of the sub-plot

        i = i + 1
plt.show()

####

# Training function

def train_for_epoch():

  # train the model
  model.train()

  # keeping track of the training losses
  training_losses = []

  for batch, targets in tqdm(training_loader):

    # moving the data to the GPU
    batch = batch.to(device)
    targets = targets.to(device)

    # set the gradient to zero
    optimizer.zero_grad()

    # forward propogration
    predictions = model(batch)

    # calculating the loss
    loss = criterion(predictions, targets)

    # backpropagate
    loss.backward()

    # updating weights
    optimizer.step()

    # update the loss in the list
    training_losses.append(loss.item())

  train_loss = np.mean(training_losses)

  return train_loss


### 

# testing function

def test():

    # put the model in the testing mode
    model.eval()

    # for tracking the losses and predictions
    test_losses = []
    test_predictions = []

    # to avoid calculating gradients
    # helps save computational resources
    with torch.no_grad():
        for batch, targets in tqdm(testing_loader):

            # Moving things to the device
            batch = batch.to(device)
            targets = targets.to(device)

            # forward propagation
            predictions = model(batch)

            # calculate the loss
            loss = criterion(predictions, targets)

            # update the losses' list
            test_losses.append(loss.item())

            # save the predictions
            test_predictions.extend(predictions.argmax(dim=1).cpu().numpy())

    # find the average test loss
    avg_test_loss = np.mean(test_losses)

    # Collect predictions into y_pred and ground truth into y_true
    y_pred = np.array(test_predictions, dtype=np.float32)
    y_true = np.array(testing_data.targets, dtype=np.float32)

    # Calculate accuracy as the average number of times y_true == y_pred
    accuracy = np.mean([y_pred[i] == y_true[i] for i in range(len(y_true))])

    return avg_test_loss, accuracy


### 

# final training

def train(first_epoch, num_epochs):

    train_losses = []
    test_losses = []
    epoch_num = []

    for epoch in range(first_epoch, first_epoch + num_epochs):
        # tracking the epoch
        epoch_num.append(epoch)

        # training for each epoch
        train_loss = train_for_epoch()
        train_losses.append(train_loss)

        # test the model
        test_loss, test_accuracy = test()

        test_losses.append(test_loss)

        print(f'[{epoch:03d}] train loss: {train_loss:04f}',
                f'test loss: {test_loss:04f}',
                f'test accuracy: {test_accuracy:04f}')

        print('')

    return epoch_num, train_losses, test_losses

### 

# Define Model

class MLP(nn.Module):

    def __init__(self, input_size=32*32*3, output_size=10):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

        return x


#### 

# Creating an instance of the model
model = MLP()

# Shift the model to the GPU
model.to(device)

###

# Initialize the optimizer and losses
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)  #optim.Adam(model.parameters(), lr=0.001)

####



def train(first_epoch, num_epochs):

    train_losses = []
    test_losses = []
    epoch_num = []

    for epoch in range(first_epoch, first_epoch + num_epochs):
        # tracking the epoch
        epoch_num.append(epoch)

        # training for each epoch
        train_loss = train_for_epoch()
        train_losses.append(train_loss)

        # test the model
        test_loss, test_accuracy = test()

        test_losses.append(test_loss)

        print(f'[{epoch:03d}] train loss: {train_loss:04f}',
                f'test loss: {test_loss:04f}',
                f'test accuracy: {test_accuracy:04f}')

        print('')

    return epoch_num, train_losses, test_losses

epochs, train_losses, test_losses = train(1, 25)# final training
###
