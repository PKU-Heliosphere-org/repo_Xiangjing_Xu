import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim import Adam
import numpy as np
from tqdm.notebook import tqdm
import os
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# 网络骨干
class WaveNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.regressor = nn.Sequential(
            nn.Linear(128*8*8, 2048),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.output_dim)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

# 损失函数，采用MSE损失
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.unbind(dim=1)
        target = target.unbind(dim=1)

        # 计算总损失
        total_loss = 0.0
        for p, t in zip(pred, target):
            total_loss += torch.nn.functional.mse_loss(p, t)

        return total_loss

# 数据集
class WaveDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data[idx]).unsqueeze(0).float()
        target_tensor = torch.tensor(self.target[idx]).float()
        return data_tensor, target_tensor

# 模型训练、验证、预测
# 默认参数是A T life sin(phase) cos(phase) begin_x begin_y
class WaveRegress:
    def __init__(self, data, output_param=None, target=None, lr=1e-3, load_path=None):
        if output_param is not None:
            self.output_param = np.array(output_param).astype(bool)
        else:
            self.output_param = np.array([1, 1, 1, 1, 1, 1, 1]).astype(bool)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = WaveNet(np.sum(self.output_param)).to(self.device)
        self.criterion = CombinedLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.data = data
        self.target = target[:, self.output_param]

        self.epoch = -1
        self.best_val_loss = np.inf

        if load_path is not None:
            param_dict = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(param_dict['model_state_dict'])
            self.optimizer.load_state_dict(param_dict['optimizer_state_dict'])
            self.epoch = param_dict['epoch']
            self.best_val_loss = param_dict['best_val_loss']

    def split_dataset(self, batch_size=32, train=0.8):
        if self.target is None:
            raise ValueError("Target can't be None")
        dataset = WaveDataset(self.data, self.target)
        train_size = int(train * len(dataset))
        val_size = len(dataset) - train_size
        if 0 < train_size < len(dataset):
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            return train_loader, val_loader
        elif train_size == 0:
            val_loader = DataLoader(dataset, batch_size=batch_size)
            return None, val_loader
        elif train_size == len(dataset):
            train_loader = DataLoader(dataset, batch_size=batch_size)
            return train_loader, None
        else:
            raise ValueError("size of dataset for train or validate is wrong")

    def train(self, epochs=50, batch_size=32, train=0.8, save_dir='./pytorch/', model_name='Wave_Regress_Model.pth'):
        if self.target is None:
            raise ValueError("Target can't be None")
        os.makedirs(save_dir, exist_ok=True)
        train_loader, val_loader = self.split_dataset(batch_size=batch_size, train=train)
        train_losses = []
        val_losses = []

        for _ in tqdm(range(epochs), desc="training"):
            self.epoch += 1
            self.model.train()
            train_loss = 0.0
            # 训练阶段
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            # 验证阶段
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            # 保存验证集上表现最佳的模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                }, os.path.join(save_dir, model_name))

        fig, ax = plt.subplots()
        ax.clear()
        ax.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
        ax.plot(range(1, epochs + 1), val_losses, label='Val Loss', marker='o')
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid()
        plt.show()

    def predict(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        if data.ndim == 3:
            data = data.unsqueeze(1)
        elif data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("Wrong data dimension")

        self.model.to(self.device)
        self.model.eval()
        data = data.to(self.device)


        with torch.no_grad():
            predictions = self.model(data)

        return predictions.cpu().numpy()

def simulation(x, t, dx=120, dt=3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # life_span = {'mean': 60, 'std': 30} # s
    life_span = {'mean': 90, 'std': 30} # s
    brightness = {'mean': 0.2, 'std': 3.85}
    # period = {'min': 60, 'max': 120} # s
    period = {'min': 20, 'max': 60} # s
    amplitude = {'mean': 21, 'std': 3} # km/s
    width = 9
    noise = {'mean': 0, 'std': 0.66}

    x_vect = np.arange(0, x, dx)
    t_vect = np.arange(0, t, dt)

    space_time = np.zeros((len(t_vect), len(x_vect)))

    life = truncnorm.rvs(-1, 3, loc=life_span['mean'], scale=life_span['std'])
    begin_time = np.random.uniform(0, t-life)
    phase = np.random.uniform(0, 2*np.pi)
    T = np.random.uniform(period['min'], period['max'])
    I = truncnorm.rvs(1, 5, loc=brightness['mean'], scale=brightness['std'])
    speed = truncnorm.rvs(-3, 3, loc=amplitude['mean'], scale=amplitude['std'])
    # A = 0.5*np.cos(np.random.uniform(0, 0.5*np.pi))*speed*T/np.pi
    A = 0.5 * speed * T / np.pi
    begin_x = np.random.uniform(A, x-A)

    t_index = (np.arange(begin_time, begin_time+life, dt)/dt).astype(int)
    t_index = np.clip(t_index, 0, len(t_vect)-1)
    c_index = np.floor((begin_x + A*np.sin(2*np.pi * (t_index*dt-begin_time)/T + phase) + np.random.normal(0, A/5, len(t_index)))/dx).astype(int)
    c_index = np.clip(c_index, 0, len(x_vect)-1)

    for j, time in enumerate(t_index):
        c = c_index[j]
        x_index = (np.array(range(width)) - width//2 + c).astype(int)
        x_index = np.clip(x_index, 0, len(x_vect)-1)
        for index in x_index:
            space_time[time, index] = I * np.exp(-(index-c)**2 / (2 * (width/12)**2))
            # space_time[time, index] = I * np.cos(np.pi/(2*(width//2))*(index-c))

    space_time += truncnorm.rvs(-10, 10, loc=noise['mean'], scale=noise['std'], size=(len(t_vect), len(x_vect)))

    space_time = (space_time - np.mean(space_time))/np.std(space_time)

    target = np.array([A/x, T/t, life/t, np.sin(phase), np.cos(phase), c_index[0]*dx/x, t_index[0]*dt/t])

    return space_time, target

def generate_simulation(num, dx=120, dt=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    datas, targets = [], []
    for i in range(num):
        data, target = simulation(x=64*dx, t=64*dt, dx=dx, dt=dt)
        datas.append(data)
        targets.append(target)
    datas = np.array(datas)
    targets = np.array(targets)
    return datas, targets

def plot_predict(data, predict):
    length = data.shape[0]
    A = predict[0] * length
    T = predict[1] * length
    phase = np.arctan2(predict[3], predict[4])
    phase = phase if phase >= 0 else phase + 2*np.pi
    life = predict[2] * length
    x = predict[5]*length
    t = predict[6]*length

    t_vect = np.linspace(t, t+life, 100)
    begin_x = x - A*np.sin(phase)
    x_vect = begin_x + A*np.sin(phase + 2*np.pi*(t_vect-t)/T)

    with plt.ioff():
        fig, ax = plt.subplots()
        ax.imshow(data, cmap='gray', origin='lower')
        ax.plot(x_vect, t_vect, color='red')
    return fig