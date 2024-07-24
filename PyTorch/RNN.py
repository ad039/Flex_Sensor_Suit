import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

# device onfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# timing parameters
start_train_time = 3
end_train_time = 4    # min
start_test_time = 5.5
end_test_time = 6
sample_frequency = 100  # Hz

#hyper parameters
input_size = 7
sequence_len = (end_train_time - start_train_time)*sample_frequency*60
num_layers = 2
hidden_size = 5000
num_classes = 3
num_epochs = 1000
learing_rate = 0.0001
batch_size = 6000





# Circle 5.5-6 min
# Box 5-5.5 min

# import train data 
class Train_FSSData(Dataset):

    def __init__(self):
        #data loading
        xy = np.loadtxt('./PyTorch/Alex_Motion_Cap_Formatted.csv', delimiter=",", dtype=np.float32, skiprows=1)
        
        hand_centre = np.column_stack((np.mean([xy[:,6], xy[:,9], xy[:,12]], axis=0), np.mean([xy[:,7], xy[:,10], xy[:,13]], axis=0), np.mean([xy[:,8], xy[:,11], xy[:,14]], axis=0)))
        shoulder_centre = np.column_stack((np.mean([xy[:,15], xy[:,18], xy[:,21]], axis=0), np.mean([xy[:,16], xy[:,19], xy[:,22]], axis=0), np.mean([xy[:,17], xy[:,20], xy[:,23]], axis=0)))
        hand_shoulder_origin = np.subtract(hand_centre, shoulder_centre)

        self.x = torch.from_numpy(xy[round(start_train_time*sample_frequency*60):round(end_train_time*sample_frequency*60), 25:32])
        self.x = self.x.reshape(sequence_len, 1, 7).type(torch.int)
        self.y = torch.from_numpy(hand_shoulder_origin[round(start_train_time*sample_frequency*60):round(end_train_time*sample_frequency*60), :]).type(torch.int)
        print(self.x.size(), self.y.size())
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(1)


# import test data
class Test_FSSData(Dataset):

    def __init__(self):
        #data loading
        xy = np.loadtxt('./PyTorch/Alex_Motion_Cap_Formatted.csv', delimiter=",", dtype=np.float32, skiprows=1)
        
        hand_centre = np.column_stack((np.mean([xy[:,6], xy[:,9], xy[:,12]], axis=0), np.mean([xy[:,7], xy[:,10], xy[:,13]], axis=0), np.mean([xy[:,8], xy[:,11], xy[:,14]], axis=0)))
        shoulder_centre = np.column_stack((np.mean([xy[:,15], xy[:,18], xy[:,21]], axis=0), np.mean([xy[:,16], xy[:,19], xy[:,22]], axis=0), np.mean([xy[:,17], xy[:,20], xy[:,23]], axis=0)))
        hand_shoulder_origin = np.subtract(hand_centre, shoulder_centre)

        self.x = torch.from_numpy(xy[round(start_test_time*sample_frequency*60):round(end_test_time*sample_frequency*60), 25:32])
        self.x = self.x.reshape(self.x.size(0), 1, 7).type(torch.int)    
        self.y = torch.from_numpy(hand_shoulder_origin[round(start_test_time*sample_frequency*60):round(end_test_time*sample_frequency*60), :]).type(torch.int)
        print(self.x, self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(1)
    

# load the data
train_dataset = Train_FSSData()
dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = Test_FSSData()
test_x, test_y = test_dataset.x, test_dataset.y
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # x -> (batch_size, sequence_length, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        x = x.type(torch.float32)
        out, _ = self.rnn(x, h0)
        # out: batch_size, sequence_length, hidden_size
        # out: (N, 34515, 500)

        out = out[:, -1, :]
        # out: (N, hidden_size)
        out = self.fc(out)
        return(out)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# loss and optimzer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)

pbar = tqdm(total=num_epochs)

# training loop
for epoch in range(num_epochs):     
    for i , (train_x, train_y) in enumerate(dataloader):
        # original size -> [batch_size, 1, 7] (need to reshape)        # send data to GPU
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        # forward pass
        train_y_pred = model(train_x)
        loss = criterion(train_y_pred, train_y)

        # backward pass
        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()

    # add stuff to progress bar in the end
    pbar.update()
    pbar.set_postfix(loss=loss.item())
pbar.close()


# test
with torch.no_grad():
    #for test_x, test_y in test_loader:
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
    RMSE_x = math.sqrt(np.square(np.subtract(test_y_numpy[:,0],test_y_pred_numpy[:,0])).mean())
    RMSE_y = math.sqrt(np.square(np.subtract(test_y_numpy[:,1],test_y_pred_numpy[:,1])).mean())
    RMSE_z = math.sqrt(np.square(np.subtract(test_y_numpy[:,2],test_y_pred_numpy[:,2])).mean())
    print(f'RMSE in x: {RMSE_x:.4f}, RMSE in y: {RMSE_y:.4f}, RMSE in z: {RMSE_z:.4f}')

    # plot
    fig, axs = plt.subplots(1, 3)
    for j in range(3):
        axs[j].plot(test_y_numpy[:, j])
        axs[j].plot(test_y_pred_numpy[:, j])
        axs[j].plot(test_y_pred_numpy_smoothed[:, j])
        axs[j].legend(["Target", "Prediction", "Prediction Smoothed"])

    fig2 = plt.figure()
    ax2 = plt.axes(projection='3d')  
    ax2.plot3D(test_y_numpy[:,0], test_y_numpy[:,1], test_y_numpy[:,2])
    ax2.plot3D(test_y_pred_numpy[:,0], test_y_pred_numpy[:,1], test_y_pred_numpy[:,2])
    ax2.plot3D(test_y_pred_numpy_smoothed[:,0], test_y_pred_numpy_smoothed[:,1], test_y_pred_numpy_smoothed[:,2])
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_zlabel('z (mm)')
    ax2.legend(["Target", "Prediction", "Prediction Smoothed"])


    plt.show()