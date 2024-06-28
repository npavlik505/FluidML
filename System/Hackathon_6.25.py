import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import date

# If multiple GPUs are available you may need to specify the device by index 'cuda:0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Import the data
Outdoor_File = r'C:\Users\natha\OneDrive\Desktop\Train_HouseZero_Outdoor.csv'
Indoor_File = r'C:\Users\natha\OneDrive\Desktop\Train_HouseZero_Indoor.csv'

Outdoor = pd.read_csv(Outdoor_File)
Indoor = pd.read_csv(Indoor_File)

Outdoor = Outdoor.values
# print("Outdoor Data")
# print("DataType:_" + str(type(Outdoor)))
# print("Number of elements:_" + str(Outdoor.size))
# print("Dimension:_" + str(Outdoor.shape))

Indoor = Indoor.values
# print("Indoor Data")
# print("DataType:_" + str(type(Indoor)))
# print("Number of elements:_" + str(Indoor.size))
# print("Dimension:_" + str(Indoor.shape))
# print(Indoor[1,1])
# print(Indoor.shape[0])
# print(Indoor.shape[1])

def day_selector(day_input, day_identification):
    return any(day_input == day_identification)

ID_6_27 = date(2022, 6, 27)

day_6_27 = []
for rows in Indoor:
    if day_selector(rows, ID_6_27):
        np.stack(day_6_27, rows)
        
# for rows in Indoor:
#     if rows[0].

##Varaibles
RT_input_dim = 8
RT_action_dim = 1
RT_hidden_width = 5
class RTNetwork(nn.Module):
    def __init__(self, RT_input_dim, RT_output_dim, RT_hidden_width):
        super(RTNetwork, self).__init__()
        self.l1 = nn.Linear(RT_input_dim, RT_hidden_width, bias = True)
        self.sigmoid1 = nn.Sigmoid()
        self.l2 = nn.Linear(RT_hidden_width, RT_action_dim, bias = True)

    def forward(self, s): #Changed l1 and l2 from F.relu() to tanh
        l1_out = torch.tanh(self.l1(s))
        sigmoid1 = nn.Sigmoid(l1_out)
        l2_out = self.l2(sigmoid1)
        return l2_out
RoomTemp = RTNetwork().to(device)

# Define loss function to use for training
criterion = nn.CrossEntropyLoss()
# Define optimizer algorithm to use for training
optimizer = optim.Adam(RoomTemp.parameters(), lr=0.001)

# Save metrics at each epoch for plotting
epoch_train_loss_values = []
epoch_val_loss_values = []
epoch_train_acc_values = []
epoch_val_acc_values = []

# Define the number of epochs to run
epochs = 25

# Using validation accuracy as metric
best_metric = -1
best_epoch = 0

for epoch in range(epochs):
    RoomTemp.train()  # Set model to training mode

    train_losses, train_accuracies = [], []

    for data, label in train_dl:
        data, label = data.to(device), label.to(device)  # Move data to the same device as the model

        optimizer.zero_grad()  # Clear previous epoch's gradients
        output = RoomTemp(data)  # Forward pass
        loss = criterion(output, label)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Accumulate metrics
        acc = (output.argmax(dim=1) == label).float().mean().item()
        train_losses.append(loss.item())
        train_accuracies.append(acc)

    # Average metrics across all training steps
    epoch_train_loss = sum(train_losses) / len(train_losses)
    epoch_train_accuracy = sum(train_accuracies) / len(train_accuracies)

    # Save current epochs training metrics
    epoch_train_loss_values.append(epoch_train_loss)
    epoch_train_acc_values.append(epoch_train_accuracy)


















# SlabTemp (ST) Values
ST_state_dim = 'TBD'
ST_action_dim = 'TBD'
ST_hidden_width = 'TBD'
ST_max_value = 'TBD'

class STNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(STNetwork, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        #self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s): #Changed l1 and l2 from F.relu() to tanh
        s = torch.tanh(self.l1(s))
        #s = torch.tanh(self.l2(s))
        a = self.max_action * torch.tanh(self.l3(s))  # [-max,max]
        return a