import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# device onfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
input_size = 7
hidden_size = 20
num_classes = 1
num_epochs = 1000
learing_rate = 0.0001
start_train_time = 0
end_train_time = 5    # min
start_test_time = 0
end_test_time = 5
sample_frequency = 100  # Hz


# Circle 5.5-6 min
# Box 5-5.5 min

# import train data 
class Train_FSSData(Dataset):

    def __init__(self):
        #data loading
        xy = np.loadtxt('./PyTorch/Alex_Motion_Cap_Formatted.csv', delimiter=",", dtype=np.float32, skiprows=1)

        self.x = torch.from_numpy(xy[round(start_train_time*sample_frequency*60):round(end_train_time*sample_frequency*60), 25:32])
        self.y = torch.from_numpy(xy[round(start_train_time*sample_frequency*60):round(end_train_time*sample_frequency*60), [3]])

        #print(self.x, self.y)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(1)


# import test data
class Test_FSSData(Dataset):

    def __init__(self):
        #data loading
        xy = np.loadtxt('./PyTorch/Alex_Motion_Cap_Formatted.csv', delimiter=",", dtype=np.float32, skiprows=1)

        self.x = torch.from_numpy(xy[round(start_test_time*sample_frequency*60):round(end_test_time*sample_frequency*60), 25:32])
        self.y = torch.from_numpy(xy[round(start_test_time*sample_frequency*60):round(end_test_time*sample_frequency*60), [3]])
        #print(self.x, self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(1)
    

# load the data
train_dataset = Train_FSSData()
#train_x, train_y = train_dataset.x, train_dataset.y
dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=False, num_workers=0)

test_dataset = Test_FSSData()
test_x, test_y = test_dataset.x, test_dataset.y

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.batchNorm = nn.BatchNorm1d(num_classes)
        self.sigmoid = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        #out = self.lrelu(out)
        #out = self.lrelu(out)
        #out = self.tanh(out)
        #out = self.tanh(out)
        out = self.relu(out)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimzer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)

pbar = tqdm(total=num_epochs)

# training loop
for epoch in range(num_epochs):     
    for i , (train_x, train_y) in enumerate(dataloader):
        #send data to GPU
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        # forward pass
        train_y_pred = model(train_x)
        loss = criterion(train_y_pred, train_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # add stuff to progress bar in the end
    #pbar.set_description(f"Epoch [{epoch}/{num_epochs}]")
    pbar.update()
    pbar.set_postfix(loss=loss.item())


# test
with torch.no_grad():
    test_y_pred = model(test_x.to(device))

    test_y_numpy = test_y.cpu().numpy()
    test_y_pred_numpy = test_y_pred.cpu().numpy()

    # smoothing
    alpha = 0.05
    n_samples = np.size(test_y_pred_numpy, 0)
    i = 2
    test_y_pred_numpy_smoothed = np.zeros_like(test_y_pred_numpy)
    test_y_pred_numpy_smoothed[0,:] = test_y_pred_numpy[0,:]
    test_y_pred_numpy_smoothed[1,:] = test_y_pred_numpy[1,:]
    test_y_pred_numpy_smoothed[2,:] = test_y_pred_numpy[2,:]

    for i in range(2, n_samples):
        test_y_pred_numpy_smoothed[i,:] = (alpha)*test_y_pred_numpy[i,:] + (1-alpha)*test_y_pred_numpy_smoothed[i-1,:]


    # calculate RMSE
    RMSE_x = math.sqrt(np.square(np.subtract(test_y_numpy[:],test_y_pred_numpy[:])).mean())
    print(f'RMSE in x: {RMSE_x:.4f}')

    #RMSE_x = math.sqrt(np.square(np.subtract(test_y_numpy[:,0],test_y_pred_numpy[:,0])).mean())
    #RMSE_y = math.sqrt(np.square(np.subtract(test_y_numpy[:,1],test_y_pred_numpy[:,1])).mean())
    #RMSE_z = math.sqrt(np.square(np.subtract(test_y_numpy[:,2],test_y_pred_numpy[:,2])).mean())
    #print(f'RMSE in x: {RMSE_x:.4f}, RMSE in y: {RMSE_y:.4f}, RMSE in z: {RMSE_z:.4f}')

    # plot
    #fig, axs = plt.subplots(1, 3)
    #for j in range(1):
    #    axs[j].plot(test_y_numpy[:, j])
    #    axs[j].plot(test_y_pred_numpy[:, j])
    #    axs[j].plot(test_y_pred_numpy_smoothed[:, j])
    #    axs[j].legend(["Target", "Prediction", "Prediction Smoothed"])

    #fig2 = plt.figure()
    #ax2 = plt.axes(projection='3d')  
    #ax2.plot3D(test_y_numpy[:,0], test_y_numpy[:,1], test_y_numpy[:,2])
    #ax2.plot3D(test_y_pred_numpy[:,0], test_y_pred_numpy[:,1], test_y_pred_numpy[:,2])
    #ax2.plot3D(test_y_pred_numpy_smoothed[:,0], test_y_pred_numpy_smoothed[:,1], test_y_pred_numpy_smoothed[:,2])
    #ax2.set_xlabel('x (mm)')
    #ax2.set_ylabel('y (mm)')
    #ax2.set_zlabel('z (mm)')
    #ax2.legend(["Target", "Prediction", "Prediction Smoothed"])

    plt.plot(test_y_numpy)
    plt.plot(test_y_pred_numpy)
    plt.plot(test_y_pred_numpy_smoothed)
    plt.legend(["Target", "Prediction", "Prediction Smoothed"])
    plt.show()