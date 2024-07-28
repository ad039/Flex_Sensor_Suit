import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# device onfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
input_size = 7
hidden_size = 1000
num_classes = 6
num_epochs = 20000
learing_rate = 0.001
start_train_time = 0
end_train_time = 5  # min
start_test_time = 6.07
end_test_time = 6.18
sample_frequency = 100 # Hz



# Circle
# start_test_time = 6.07
#end_test_time = 6.18
# Box 5-5.5 min

#data loading
xy = np.loadtxt('./PyTorch/data/Alex_Motion_Cap_Test_Formatted_New.csv', delimiter=",", dtype=np.float32, skiprows=1)


#scaling functions
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# import train data 
class FSSData(Dataset):

    def __init__(self, dataset):
        self.n_samples = dataset.shape[0]
        #print(dataset.shape)
       
        Elbow_centre = dataset[:, 15:18]
        hand_centre = dataset[:, 12:15]
        shoulder_centre = dataset[:, 6:9]
        elbow_shoulder_origin = np.subtract(Elbow_centre, shoulder_centre)
        hand_shoulder_origin = np.subtract(hand_centre, shoulder_centre)
        
        self.x = dataset[:, 37:44]
        self.y = np.concat((hand_shoulder_origin, elbow_shoulder_origin), axis=1)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(1)


# load the data
train_dataset = FSSData(xy[round(start_train_time*sample_frequency*60):round(end_train_time*sample_frequency*60), :])
train_x, train_y = train_dataset.x, train_dataset.y
#dataloader = DataLoader(dataset=train_dataset, batch_size=34515, shuffle=True, num_workers=1)


test_dataset = FSSData(xy[round(start_test_time*sample_frequency*60):round(end_test_time*sample_frequency*60), :])
test_x, test_y = test_dataset.x, test_dataset.y


#scale the data and send to tensors
scaler_x.fit(train_x)
train_x = torch.from_numpy(scaler_x.transform(train_x))
test_x = torch.from_numpy(scaler_x.transform(test_x))

scaler_y.fit(train_y)
train_y = torch.from_numpy(scaler_y.transform(train_y))
test_y = torch.from_numpy(scaler_y.transform(test_y))


def calc_elbow_length(x, y, z):
    return torch.sqrt(x**2 + y**2 + z**2)

def calc_distance(x_0, y_0, z_0, x_1, y_1, z_1):
    return torch.sqrt((x_1-x_0)**2 + (y_1-y_0)**2 + (z_1-z_0)**2)


# find the known average distance between elbow and shoulder this will be a physical restaint
shoulder_to_elbow = torch.mean(calc_elbow_length(test_y[:,3], test_y[:,4], test_y[:,5]))

# find the known average distance between elbow and hand this will be a physical restaint
elbow_to_hand = torch.mean(calc_distance(test_y[:,3], test_y[:,4], test_y[:,5], test_y[:,0], test_y[:,1], test_y[:,2]))


# build a neural network module
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.type(torch.float32)
        out = self.l1(x)
        out = self.tanh(out)
        out = self.tanh(out)
        out = self.sig(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimzer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)

# pbar initialization
pbar = tqdm(total=num_epochs)

# training loop
for epoch in range(num_epochs):     
    #for i , (train_x, train_y) in enumerate(dataloader):
    #send data to GPU
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # forward pass
    train_y_pred = model(train_x)

    loss = torch.mean((train_y_pred - train_y)**2)
    # for elbow physics constrained model
    loss += 2*torch.mean((calc_elbow_length(train_y_pred[:,3], train_y_pred[:,4], train_y_pred[:,5])- shoulder_to_elbow)**2)

    # for hand physics constrained model
    loss += 2*torch.mean((calc_distance(train_y_pred[:,3], train_y_pred[:,4], train_y_pred[:,5], train_y_pred[:,0], train_y_pred[:,1], train_y_pred[:,2])- elbow_to_hand)**2)

    # backward pass
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # add stuff to progress bar in the end
    #pbar.set_description(f"Epoch [{epoch}/{num_epochs}]")
    pbar.update()
    pbar.set_postfix(loss=loss.item())
pbar.close()


# test
with torch.no_grad():

    # initialize test_y_pred size
    test_num_samples = test_x.size(0)
    test_y_pred = torch.zeros_like(test_y)

    #loop over test inputs like a realtime simulation
    for i in range(test_num_samples):
        test_y_pred[i, :] = model(test_x[i, :].to(device)).to(device)

    # send data to cpu, convet to numpy and transform back to mm data
    test_y_numpy = scaler_y.inverse_transform(test_y.cpu().numpy())
    test_y_pred_numpy = scaler_y.inverse_transform(test_y_pred.cpu().numpy())

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


    plot_labels = ["X", "Y", "Z"]


    # plot hand prediction
    fig, axs = plt.subplots(2, 3)
    for j in range(3):
        axs[0, j].plot(test_y_numpy[:, j])
        axs[0, j].plot(test_y_pred_numpy[:, j])
        axs[0, j].plot(test_y_pred_numpy_smoothed[:, j])
        axs[0, j].legend(["Hand Target", "Prediction", "Prediction Smoothed"])
        axs[0, j].set_title(plot_labels[j])
        axs[0, j].set_ylabel("Position (mm)")


    # plot elbow prediction
    for j in range(3):
        axs[1, j].plot(test_y_numpy[:, j+3])
        axs[1, j].plot(test_y_pred_numpy[:, j+3])
        axs[1, j].plot(test_y_pred_numpy_smoothed[:, j+3])
        axs[1, j].legend(["Elbow Target", "Prediction", "Prediction Smoothed"])
        axs[1, j].set_title(plot_labels[j])
        axs[1, j].set_ylabel("Position (mm)")


    # plot 3d hand position
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