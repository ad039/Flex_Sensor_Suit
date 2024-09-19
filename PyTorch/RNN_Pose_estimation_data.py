# Import Libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math

# trianing time
start_train_time = 0
end_train_time = 4
sample_frequency = 1/0.15 # Hz

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Prepare Dataset
# load data
dataset = pd.read_csv(r"PyTorch/data/Mediapipe_Pose_Estimation_Data.csv",dtype = np.float32)

# split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = dataset['P_x'].to_numpy()*1000
features_numpy = dataset[dataset.columns[1:8]].to_numpy()

targets_numpy = targets_numpy.reshape(-1, 1)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
targets_numpy = scaler_y.fit_transform(targets_numpy)
features_numpy = scaler_x.fit_transform(features_numpy)


# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy[round(start_train_time*sample_frequency*60):round(end_train_time*sample_frequency*60), :],
                                                                             targets_numpy[round(start_train_time*sample_frequency*60):round(end_train_time*sample_frequency*60), :],
                                                                             test_size = 0.2,
                                                                             random_state = 42, 
                                                                             shuffle=True) 

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train)

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test)


# Create RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
        #print(h0.shape)
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out

# batch_size, epoch and iteration
batch_size = 5
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = TensorDataset(featuresTrain,targetsTrain)
test = TensorDataset(featuresTest,targetsTest)

# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)
    
# Create RNN
input_dim = 7    # input dimension
hidden_dim = 10000  # hidden layer dimension
layer_dim = 3     # number of hidden layers
output_dim = 1   # output dimension

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)

# Cross Entropy Loss 
error = nn.SmoothL1Loss()

# SGD Optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



# Training 
seq_dim = 1  
loss_list = []
iteration_list = []
accuracy_list = []
count = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        train  = Variable(images.view(-1, seq_dim, input_dim))
        labels = Variable(labels)

        #send to device
        train = train.to(device)
        labels = labels.to(device)
            
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        #print('output', outputs, labels, train)

        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 250 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                images = Variable(images.view(-1, seq_dim, input_dim))
                
                #send to device
                labels = labels.to(device)

                # Forward propagation
                outputs = model(images.to(device))
                #print(outputs)


                # calculate RMSE
                RMSE_x = torch.mean(torch.sqrt(torch.square(torch.subtract(labels,outputs)))).cpu().detach().numpy()
                
                # Total number of labels
                total += labels.size(0)
                
                correct += RMSE_x
            
            RMSE = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data.cpu())
            iteration_list.append(count)
            accuracy_list.append(RMSE)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  RMSE: {}'.format(count, loss.item(), RMSE))

# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("RNN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("RNN: Accuracy vs Number of iteration")
#plt.savefig('graph.png')
plt.show()





## Test ##

start_test_time = 3
end_test_time = 3.5
sample_frequency = 1/0.15 # Hz

## Data 1
# Circle
# start_test_time = 4.15
# end_test_time = 4.3
# Box 3.5-4

## Dataset 2
# circle: 3 - 3.5

## Dataset 3
# circle: 3 - 3.5


test_x = torch.from_numpy(features_numpy[round(start_test_time*sample_frequency*60):round(end_test_time*sample_frequency*60), :]).to(device)
test_y = torch.from_numpy(targets_numpy[round(start_test_time*sample_frequency*60):round(end_test_time*sample_frequency*60), :]).to(device)

print(test_x.shape)
# test
with torch.no_grad():
    test_num_samples = test_x.size(0)
    test_y_pred = torch.zeros_like(test_y)
    
    for i in range(round(test_num_samples/batch_size)):
        test_y_pred[batch_size*i:batch_size*(i+1), :] = model(test_x[batch_size*i:batch_size*(i+1), :].reshape(1, batch_size, 7)).to(device)

    print(test_y.cpu().numpy().shape)
    test_y_numpy = scaler_y.inverse_transform(test_y.cpu().numpy())
    test_y_pred_numpy = scaler_y.inverse_transform(test_y_pred.cpu().numpy())

    # smoothing
    alpha = 0.5
    n_samples = np.size(test_y_pred_numpy, 0)
    i = 2
    test_y_pred_numpy_smoothed = np.zeros_like(test_y_pred_numpy)
    test_y_pred_numpy_smoothed[0,:] = test_y_pred_numpy[0,:]
    test_y_pred_numpy_smoothed[1,:] = test_y_pred_numpy[1,:]
    test_y_pred_numpy_smoothed[2,:] = test_y_pred_numpy[2,:]

    for i in range(2, n_samples):
        test_y_pred_numpy_smoothed[i,:] = (alpha)*test_y_pred_numpy[i,:] + (1-alpha)*test_y_pred_numpy_smoothed[i-1,:]

    
    test_time = np.arange(test_y_numpy.shape[0])/sample_frequency   
    test_time = test_time.reshape(test_y_numpy.shape[0])

# calculate RMSE
    RMSE_x = math.sqrt(np.square(np.subtract(test_y_numpy[:,0],test_y_pred_numpy[:,0])).mean())
    MAX_x = np.max(np.subtract(test_y_numpy[:,0],test_y_pred_numpy[:,0]))

    print(f'RMSE: {RMSE_x:.4f}, Max Error: {MAX_x:.4f}')

    # plot
    plt.plot(test_time, test_y_numpy)
    plt.plot(test_time, test_y_pred_numpy)
    plt.plot(test_time, test_y_pred_numpy_smoothed)
    plt.legend(["Target", "Prediction", "Prediction Smoothed"])

    plt.title(f'RMSE: {RMSE_x:.4f}, Max Error: {MAX_x:.4f}')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (mm)")

    plt.show()