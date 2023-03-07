import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from scipy.io import loadmat
import numpy as np

# data load
total_data = np.load(r'/data/home/scv6730/run/new_code/total_data.npy')
# total_data = np.load(r'/data/home/scv6730/run/new_code/total_data_noise_10.npy')


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def creat_dataset(dataset, look_back):
    data_x = []
    data_y = []
    for i in range(len(dataset) - look_back*2):
        data_x.append(dataset[i:i+look_back])
        # data_y.append(dataset[i+look_back])
        data_y.append(dataset[i+look_back:i+2*look_back])    # 输入dim = 输出dim
    return np.asarray(data_x), np.asarray(data_y)


# LSTM_model
class Lstm_wind_pred(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
    # def __init__(self, input_size, hidden_size, num_layers):
        super(Lstm_wind_pred, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size   # output_size = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.5)
        self.fc0 = nn.Linear(self.hidden_size, self.output_size)
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x1, _ = self.lstm(x)
        a, b, c = x1.shape
        # print('a,b,c:', a, b, c)    # (Batch, look_back, hidden_size)
        # print('x1', x1.view(-1, c).shape)
        out0, out1, out2, out3, out4 = self.fc0(x1.view(-1, c)), self.fc1(x1.view(-1, c)), self.fc2(x1.view(-1, c)), self.fc3(x1.view(-1, c)), self.fc4(x1.view(-1, c))

        # print('out1', out1.shape)
        out0, out1, out2, out3, out4 = out0.view(a, b, -1), out1.view(a, b, -1), out2.view(a, b, -1), out3.view(a, b, -1), out4.view(a, b, -1)
        # print('out1', out1.shape)   # (Batch, look_back, out_dim)
        out = torch.stack((out0, out1, out2, out3, out4), dim=0)
        # print('out', out.shape)
        out = out.permute(3, 1, 2, 0).squeeze()
        # print('out', out.shape)

        return out


# train
def train(model, criterion, optimizer, train_data):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(train_data):
        src, tgt = batch
        # print('src_t.size0:', src.shape)
        src = src.to(device)
        tgt = tgt.to(device)

        src_t = src[:, :, 0].requires_grad_()
        src_SSTA= src[:, :, 1].requires_grad_()
        src_hA= src[:, :, 2].requires_grad_()
        src_TAO1= src[:, :, 3].requires_grad_()
        src_TAO2= src[:, :, 4].requires_grad_()
        src = torch.stack((src_t, src_SSTA, src_hA, src_TAO1, src_TAO2), dim=0)
        # print('src_t.size1:', src.shape)

        src = src.permute(1, 2, 0).requires_grad_()
        # print('src_t.size1:', src.shape)

        with torch.backends.cudnn.flags(enabled=False):
            out = model(src)
        # print('out', out.shape)

        SSTA_pred = out[:, :, 1].requires_grad_()
        HA_pred = out[:, :, 2].requires_grad_()
        TAO1_pred = out[:, :, 3].requires_grad_()
        TAO2_pred = out[:, :, 4].requires_grad_()

        tgt_SSTA = tgt[:, :, 1]
        tgt_HA = tgt[:, :, 2]
        tgt_TAO1 = tgt[:, :, 3]
        tgt_TAO2 = tgt[:, :, 4]

        T_t = torch.autograd.grad(SSTA_pred, src_t, grad_outputs=torch.ones_like(SSTA_pred), retain_graph=True, create_graph=True)[0]
        h_t = torch.autograd.grad(HA_pred, src_t, grad_outputs=torch.ones_like(HA_pred), retain_graph=True, create_graph=True)[0]   #  allow_unused=True

        f_4 = T_t + 1.8 * SSTA_pred - 1.125 * HA_pred + 1.2 * (SSTA_pred ** 3)
        g_4 = h_t + 27 * SSTA_pred + 5 * HA_pred

        # f_4 = -1.8 * SSTA_pred + 1.125 * HA_pred - 1.2 * (SSTA_pred ** 3)
        # g_4 = -27 * SSTA_pred - 5 * HA_pred

        # loss1 = criterion(SSTA_pred, tgt_SSTA) + criterion(HA_pred, tgt_HA)
        loss1 = criterion(SSTA_pred, tgt_SSTA) + criterion(HA_pred, tgt_HA) + criterion(TAO1_pred, tgt_TAO1) + criterion(TAO2_pred, tgt_TAO2)

        loss2 = torch.mean(torch.pow((f_4), 2)) + torch.mean(torch.pow((g_4), 2))
        # loss3 = criterion(T_t, f_4) + criterion(h_t, g_4)
        lam = 20

        loss = lam * loss1 + loss2
        # loss = loss3
        loss = loss.to(device)

        # loss = criterion(out, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_data)


def main():
    look_back = 6

    input_size = 5
    hidden_size = 10
    num_layers = 2
    output_size = 1

    batch_size = 64
    epochs = 2000
    epoch_loss_all = []
    best_loss = 200

    input_tensor = total_data
    print('input_tensor', input_tensor.shape)

    dataX, dataY = creat_dataset(input_tensor, look_back=look_back)
    print('dataX', dataX.shape)
    train_size = int(len(dataX) * 0.9)
    # train_size = int(len(dataX))

    # train set
    train_x = dataX[:train_size]
    print('train_x', train_x.shape)
    train_y = dataY[:train_size]
    print('train_y', train_y.shape)

    train_x = torch.from_numpy(train_x.astype(np.float32))
    train_y = torch.from_numpy(train_y.astype(np.float32))

    train_data =torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1)

    Wind_pred = Lstm_wind_pred(input_size, hidden_size, num_layers, output_size)
    Wind_pred = Wind_pred.to(device)

    optimizer = torch.optim.Adam(Wind_pred.parameters(), lr=0.003)
    criterion = nn.MSELoss()
    for i in range(epochs):
        epoch_loss = train(Wind_pred, criterion, optimizer, train_loader)
        epoch_loss_all.append(epoch_loss)
        print("epoch: {} PILSTM4_6_1_20 train loss: {}".format(i, epoch_loss))
        if epoch_loss < best_loss:
            # best_loss = epoch_loss
            model_name = r'/data/home/scv6730/run/model/LSTM/4/PILSTM4_6_1_20_model_{0:.5f}.pt'.format(epoch_loss)
            torch.save(Wind_pred.state_dict(), model_name)

if __name__ == "__main__":
    main()





