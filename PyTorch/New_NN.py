import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import math

# device onfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Train_Data = pd.read_csv("PyTorch/Alex_Motion_Cap_Formatted.csv")

FEATURE_COLUMNS = Train_Data.columns.tolist()[24:]
LABEL_COLUMNS = Train_Data.columns.tolist()[12]

FEATURE_COLUMNS.append(LABEL_COLUMNS)

data = Train_Data[FEATURE_COLUMNS]

from copy import deepcopy as dc

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Timestamp', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Hand_Top_X(t-{i})'] = df['Hand_Top_X'].shift(i)

    df.dropna(inplace=True)

    return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(data, lookback)

shifted_df_as_np = shifted_df.to_numpy()


# hyper parameters
input_size = 7
hidden_size = 10000
num_classes = 3
num_epochs = 10000
learing_rate = 0.0001
batch_size = 10

# timing parameters
start_train_time = 0
end_train_time = 4    # min
start_test_time = 5.5
end_test_time = 6
sample_frequency = 100 # Hz


# import train data 
class FSSData(Dataset):

    def __init__(self, train_dataset):

        self.x = torch.from_numpy(np.concat((train_dataset[:, 0:7], train_dataset[:, 8:]), axis=1)).type(torch.float32)
        self.x = self.x.reshape(-1, self.x.size(1), 1)
        self.y = torch.from_numpy(train_dataset[:, [7]]).type(torch.float32)
        self.y = self.y.reshape(-1, 1)
        print(self.x.shape, self.y.shape)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(1)
    
# load the data
train_dataset = FSSData(shifted_df_as_np[round(start_train_time*sample_frequency*60):round(end_train_time*sample_frequency*60), :])
test_dataset = FSSData(shifted_df_as_np[round(start_test_time*sample_frequency*60):round(end_test_time*sample_frequency*60), :])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.type(torch.float32)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(1, hidden_size, 1)
model.to(device)

def train_one_epoch():
    model.train(True)
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    return avg_loss_across_batches


learning_rate = 0.001
num_epochs = 100
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

pbar = tqdm(total=num_epochs)

for epoch in range(num_epochs):
    train_one_epoch()
    val = validate_one_epoch()

    pbar.update()
    pbar.set_postfix(loss=val)

pbar.close()
model.train(False)

with torch.no_grad():
    test_y_numpy = np.zeros([1,1])
    test_y_pred_numpy = np.zeros([1,1])

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        test_y_pred = model(x_batch)

        test_y_numpy = np.concat((test_y_numpy, y_batch.cpu().numpy()), axis=0)
        test_y_pred_numpy = np.concat((test_y_pred_numpy, test_y_pred.cpu().numpy()))
        print(test_y_numpy.shape)

    # calculate RMSE
#    RMSE_x = math.sqrt(np.square(np.subtract(test_y_numpy[:,0],test_y_pred_numpy[:,0])).mean())
#    RMSE_y = math.sqrt(np.square(np.subtract(test_y_numpy[:,1],test_y_pred_numpy[:,1])).mean())
#    RMSE_z = math.sqrt(np.square(np.subtract(test_y_numpy[:,2],test_y_pred_numpy[:,2])).mean())
#    print(f'RMSE in x: {RMSE_x:.4f}, RMSE in y: {RMSE_y:.4f}, RMSE in z: {RMSE_z:.4f}')

    # plot
    plt.plot(test_y_numpy)
    plt.plot(test_y_pred_numpy)
        
    plt.legend(["Target", "Prediction", "Prediction Smoothed"])      
    plt.show()






