import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network Architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.1):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_prob)  

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)  
        x = self.relu(self.layer2(x))
        x = self.dropout(x)  
        x = self.relu(self.layer3(x))
        x = self.dropout(x)  
        x = self.output(x)
        return self.sigmoid(x)
    

def prepare_data(X_train=None, y_train=None, X_val=None, y_val=None, X_test_scaled=None, test=False):
    """
    Prepares Data Loaders for Training , Testing and Validation.
    """
    
    if test:
        testset = TensorDataset(torch.from_numpy(X_test_scaled).float())  # Test set has no labels
        testloader = DataLoader(testset, batch_size=32, shuffle=False)

        return testloader
    
    else:
        # Convert training and validation data to TensorDatasets
        trainset = TensorDataset(torch.from_numpy(X_train).float(), 
                                torch.from_numpy(np.array(y_train)).float())
        valset = TensorDataset(torch.from_numpy(X_val).float(), 
                            torch.from_numpy(np.array(y_val)).float())
        
        # Create Dataloaders
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        valloader = DataLoader(valset, batch_size=32, shuffle=False)
        
        return trainloader, valloader


def train_network(X_train, y_train, X_val, y_val):
    """
    Trains the Neural Network using weighted decay with Adam optimizer.
    """
    trainloader, valloader = prepare_data(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val) 
    input_dim = X_train.shape[1]
    net = NeuralNetwork(input_dim)

    # Define loss and optimizer
    cost_fn = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-5)

    num_iter = 200
    net = net.to(device)

    for epoch in range(num_iter):
        net.train()
        running_loss = 0.0
        true_labels = []
        pred_probs = []

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs).view(-1)  # Forward pass
            loss = cost_fn(outputs, labels)  # Loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            true_labels.extend(labels.cpu().numpy())
            pred_probs.extend(outputs.cpu().detach().numpy())

        train_auroc = roc_auc_score(true_labels, pred_probs)

        # Validate
        net.eval()
        val_true = []
        val_pred = []
        with torch.no_grad():
            for val_inputs, val_labels in valloader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = net(val_inputs).view(-1)
                val_true.extend(val_labels.cpu().numpy())
                val_pred.extend(val_outputs.cpu().numpy())

        val_auroc = roc_auc_score(val_true, val_pred)
        print(f'Epoch [{epoch+1}/{num_iter}], Loss: {running_loss:.4f}, Train AUROC: {train_auroc:.4f}, Val AUROC: {val_auroc:.4f}')
    
    return net


def predict_test(net, X_test_scaled):
    """
    Sets the trained neural network in eval mode to generate predictions on Test Data.
    """
    testloader = prepare_data(X_test_scaled=X_test_scaled, test=True)
    net.eval()
    test_preds = []

    with torch.no_grad():
        for inputs in testloader:
            inputs = inputs[0].to(device)  # Testloader yields inputs only (no labels)
            outputs = net(inputs).view(-1)
            test_preds.extend(outputs.cpu().numpy())

    return np.array(test_preds)

