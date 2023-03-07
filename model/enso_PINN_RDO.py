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

input_tensor = total_data
# print('input_tensor', input_tensor.shape)


# 时间步data构造函数
def creat_dataset(dataset, look_back):
    data_x = []
    data_y = []
    for i in range(len(dataset) - look_back*2):
        data_x.append(dataset[i:i+look_back])
        # data_y.append(dataset[i+look_back])
        data_y.append(dataset[i+look_back:i+2*look_back])    # 输入dim = 输出dim
    return np.asarray(data_x), np.asarray(data_y)


# Neural Network
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 4)
        )

    def forward(self, x):
        # print('x.shape:', x.shape)
        a, b, c = x.shape    # batch, timesteps, indim
        # x = x.view(-1, c)
        out = self.net(x.view(-1, c))
        out = out.view(a, b, -1)
        # print('out.shape', out.shape)

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
        src_SSTA = src[:, :, 1].requires_grad_()
        src_hA = src[:, :, 2].requires_grad_()
        src_TAO1 = src[:, :, 3].requires_grad_()
        src_TAO2 = src[:, :, 4].requires_grad_()
        src = torch.stack((src_t, src_SSTA, src_hA, src_TAO1, src_TAO2), dim=0)
        # print('src_t.size1:', src.shape)

        src = src.permute(1, 2, 0).requires_grad_()
        # print('src_t.size1:', src.shape)

        with torch.backends.cudnn.flags(enabled=False):
            out = model(src)
        # print('out', out.shape)

        SSTA_pred = out[:, :, 0].requires_grad_()
        HA_pred = out[:, :, 1].requires_grad_()
        TAO1_pred = out[:, :, 2].requires_grad_()
        TAO2_pred = out[:, :, 3].requires_grad_()

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
        lam = 10
        loss = lam * loss1 + loss2
        # loss = loss3
        loss = loss.to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_data)

def main():
    look_back = 6

    batch_size = 64
    epochs = 20000
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

    # train_x = train_x.reshape(-1, look_back, input_size)
    # print('train_x', train_x.shape)
    # train_y = train_y.reshape(-1, 1, input_size)
    # print('train_y', train_y.shape)


    train_x = torch.from_numpy(train_x.astype(np.float32))
    train_y = torch.from_numpy(train_y.astype(np.float32))

    train_data =torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=1)

    U = MLP()
    U = U.to(device)

    optimizer = torch.optim.Adam(U.parameters(), lr=0.003)
    criterion = nn.MSELoss()
    for i in range(epochs):
        epoch_loss = train(U, criterion, optimizer, train_loader)
        epoch_loss_all.append(epoch_loss)
        print("epoch: {} PINN4_6_1_20 train loss: {}".format(i, epoch_loss))
        if epoch_loss < best_loss:
            # best_loss = epoch_loss
            model_name = r'/data/home/scv6730/run/model/NN/4/PINN4_6_1_20_model_{0:.5f}.pt'.format(epoch_loss)
            torch.save(U.state_dict(), model_name)

if __name__ == "__main__":
    main()





