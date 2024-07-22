import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import math

# device onfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
input_size = 7
hidden_size = 10000
num_classes = 3
num_epochs = 1000
learing_rate = 0.1
start_train_time = 3
end_train_time = 4        # min
start_test_time = 5.5
end_test_time = 6
sample_frequency = 100  # Hz


# Cirlce 5.5-6min
# Box 

# import train data 
class Train_FSSData(Dataset):

    def __init__(self):
        #data loading
        xy = np.loadtxt('./PyTorch/Alex_Motion_Cap_Formatted.csv', delimiter=",", dtype=np.float32, skiprows=1)
        
        # smoothing
        alpha = 0.1
        n_samples = np.size(xy, 0)
        i = 2
        for i in range(n_samples):
            xy[i,7:10] = (alpha)*xy[i,7:10] + (1-alpha)*xy[i-1,7:10]
        
        self.x = torch.from_numpy(xy[round(start_train_time*sample_frequency*60):round(end_train_time*sample_frequency*60), 25:32]).to(device)
        self.y = torch.from_numpy(xy[round(start_train_time*sample_frequency*60):round(end_train_time*sample_frequency*60), 12:15]).to(device)
        #print(self.x, self.y)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        #dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        #len(dataset)
        return self.n_samples


# import test data
class Test_FSSData(Dataset):

    def __init__(self):
        #data loading
        xy = np.loadtxt('./PyTorch/Alex_Motion_Cap_Formatted.csv', delimiter=",", dtype=np.float32, skiprows=1)

        # smoothing
        alpha = 0.1
        n_samples = np.size(xy, 0)
        i = 2
        for i in range(n_samples):
            xy[i,7:10] = (alpha)*xy[i,7:10] + (1-alpha)*xy[i-1,7:10]


        self.x = torch.from_numpy(xy[round(start_test_time*sample_frequency*60):round(end_test_time*sample_frequency*60), 25:32]).to(device)
        self.y = torch.from_numpy(xy[round(start_test_time*sample_frequency*60):round(end_test_time*sample_frequency*60), 12:15]).to(device)
        #print(self.x, self.y)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        #dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        #len(dataset)
        return self.n_samples
    

# load the data
train_dataset = Train_FSSData()
train_x, train_y = train_dataset.x, train_dataset.y

test_dataset = Test_FSSData()
test_x, test_y = test_dataset.x, test_dataset.y

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        #out = self.lrelu(out)
        #out = self.lrelu(out)
        out = self.tanh(out)
        out = self.tanh(out)
        #out = self.relu(out)
        #out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimzer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)


# training loop
for epoch in range(num_epochs):
    
    # forward pass
    train_y_pred = model(train_x)
    loss = criterion(train_y_pred, train_y)

    # backward pass
    loss.backward()
    
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % (num_epochs/100) == 0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')


# test
with torch.no_grad():
    test_y_pred = model(test_x)

    test_y_numpy = test_y.cpu().numpy()
    test_y_pred_numpy = test_y_pred.cpu().numpy()

    # smoothing
    alpha = 0.3
    n_samples = np.size(test_y_pred_numpy, 0)
    i = 2
    for i in range(n_samples):
        test_y_pred_numpy[i,:] = (alpha)*test_y_pred_numpy[i,:] + (1-alpha)*test_y_pred_numpy[i-1,:]


    # calculate RMSE
    RMSE_x = math.sqrt(np.square(np.subtract(test_y_numpy[:,0],test_y_pred_numpy[:,0])).mean())
    RMSE_y = math.sqrt(np.square(np.subtract(test_y_numpy[:,1],test_y_pred_numpy[:,1])).mean())
    RMSE_z = math.sqrt(np.square(np.subtract(test_y_numpy[:,2],test_y_pred_numpy[:,2])).mean())
    print(f'RMSE in x: {RMSE_x:.4f}, RMSE in y: {RMSE_y:.4f}, RMSE in z: {RMSE_z:.4f}')

    # plot
    fig, axs = plt.subplots(1, 3)
    for j in range(3):
        axs[j].plot(test_y_numpy[:, j])
        axs[j].plot(test_y_pred_numpy[:, j])
        axs[j].legend(["Target", "Prediction"])

    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')  
    ax2.plot3D(test_y_numpy[:,0], test_y_numpy[:,1], test_y_numpy[:,2])
    ax2.plot3D(test_y_pred_numpy[:,0], test_y_pred_numpy[:,1], test_y_pred_numpy[:,2])
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_zlabel('z (mm)')
    ax2.legend(["Target", "Prediction"])


    plt.show()
