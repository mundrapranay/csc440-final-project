import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time 

DATA_LOC = './model_data/{0}/{1}.csv'
METRIC_FILE = './metric_files/{0}_{1}_{2}_{3}.txt'
# Define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Define the model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_model(params):
    # Initialize the model
    input_size = 182
    hidden_size = 128
    num_classes = 4
    model = Net(input_size, hidden_size, num_classes)

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load the data using pandas
    train = pd.read_csv(params.train_datafile)
    # train, test = train_test_split(data, test_size=0.3)

    train_labels = train['label'].to_numpy()
    # train_labels = torch.from_numpy(train_labels)
    train_data = train.drop('label', axis=1).to_numpy()

    # Initialize the custom dataloader
    train_dataset = CustomDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train the model
    start_time = time.time()
    num_epochs = 100
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = []
        for data, labels in train_dataloader:
            # Forward pass
            outputs = model(data.float())
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record the loss
            epoch_loss.append(loss.item())
        losses.append(sum(epoch_loss) / len(epoch_loss))
    
    end_time = time.time()
    plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), losses)
    plt.show()
    plt.savefig('./figures/model_loss_{0}_{1}.png'.format(params.train_league, params.train_season))
    plt.cla()
    metric_file = METRIC_FILE.format(params.train_season, params.train_league, params.test_season, params.test_league)
    with open(metric_file, 'a') as f:
        f.write('Time Taken : {0}\n'.format(end_time - start_time))
    f.close()
    # Save the model 

    return model 

def test_model(model, params):
    # Test the model
    model.eval()
    test = pd.read_csv(params.test_datafile)
    test_labels = test['label'].to_numpy()
    test_data = test.drop('label', axis=1).to_numpy()

    # Initialize the custom dataloader
    test_dataset = CustomDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    class_wise_accs = np.zeros(4)
    with torch.no_grad():
        correct = 0
        total = 0
        class_wise_corrcet = [0] * 4
        class_wise_samples = [0] * 4
        for data, labels in test_dataloader:
            outputs = model(data.float())
            _, predicted = torch.max(outputs.data, 1)
            for c in range(4):
                class_wise_corrcet[c] += torch.sum((predicted == labels) * (labels == c))
                class_wise_samples[c] += torch.sum(labels == c)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        for c in range(4):
            class_wise_accs[c] = float(class_wise_corrcet[c]) / float(class_wise_samples[c]) * 100
        
        metric_file = METRIC_FILE.format(params.train_season, params.train_league, params.test_season, params.test_league)
        with open(metric_file, 'a') as f:
            f.write('Model Trained on {0}:{1}\t Tested on {2}:{3}\n'.format(params.train_league, params.train_season, params.test_league, params.test_season))
            f.write('Accuracy of the model on the test data: {} %\n'.format(100 * correct / total))
            f.write('Class Wise Test Accuracy\n')
            for idx, value in enumerate(class_wise_accs):
                f.write('Class:{0}\tAcc:{1}\n'.format(idx, value))
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_league', type=str, default='premier_league')
    parser.add_argument('--train_season', type=str, default='2019-2020')
    parser.add_argument('--test_league', type=str, default='bundesliga')
    parser.add_argument('--test_season', type=str, default='2019-2020')

    params = parser.parse_args()
    # print(params.train_league)
    params.train_datafile = DATA_LOC.format(params.train_season, params.train_league)
    params.test_datafile = DATA_LOC.format(params.test_season, params.test_league)
    
    model = train_model(params)
    test_model(model, params)
