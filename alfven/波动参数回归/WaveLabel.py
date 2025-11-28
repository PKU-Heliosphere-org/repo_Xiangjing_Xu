import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
from scipy.stats import truncnorm
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import image as img
from sklearn.model_selection import train_test_split
import matplotlib.patches as patches

# 产生模拟训练数据
def simulation(x, t, dx=120, dt=3, num_range=(100, 200), seed=None):
    if seed is not None:
        np.random.seed(seed)

    life_span = {'mean': 60, 'std': 30} # s
    brightness = {'mean': 0.2, 'std': 3.85}
    period = {'min': 60, 'max': 120} # s
    amplitude = {'mean': 121, 'std': 3} # km/s
    width = 9
    noise = {'mean': 0, 'std': 0.66}

    x_vect = np.arange(0, x, dx)
    t_vect = np.arange(0, t, dt)

    space_time = np.zeros((len(t_vect), len(x_vect)))

    num = int(np.random.uniform(num_range[0], num_range[-1]))
    boxes = []
    labels = []

    for i in range(num):
        life = truncnorm.rvs(-1, 3, loc=life_span['mean'], scale=life_span['std'])
        phase = np.random.uniform(0, 2*np.pi)
        T = np.random.uniform(period['min'], period['max'])
        I = truncnorm.rvs(1, 5, loc=brightness['mean'], scale=brightness['std'])
        speed = truncnorm.rvs(-3, 3, loc=amplitude['mean'], scale=amplitude['std'])
        A = 0.5*np.cos(np.random.uniform(0, 0.5*np.pi))*speed*T/np.pi
        begin_time = np.random.uniform(0, t-int(life))
        begin_x = np.random.uniform(int(A), x-int(A))

        t_index = (np.arange(begin_time, begin_time+life, dt)/dt).astype(int)
        t_index = np.clip(t_index, 0, len(t_vect)-1)
        c_index = np.floor((begin_x + A*np.sin(2*np.pi * t_index*dt/T + phase) + np.random.normal(0, A/5, len(t_index)))/dx).astype(int)
        c_index = np.clip(c_index, 0, len(x_vect)-1)

        x_indexes = []

        for j, time in enumerate(t_index):
            c = c_index[j]
            x_index = (np.array(range(width)) - width//2 + c).astype(int)
            x_index = np.clip(x_index, 0, len(x_vect)-1)
            x_indexes.extend(x_index)
            for index in x_index:
                space_time[time, index] = I * np.exp(-(index-c)**2 / (2 * (width/12)**2))
        x_indexes = np.array(x_indexes)
        box = [np.min(x_indexes), np.min(t_index), np.max(x_indexes), np.max(t_index)]
        boxes.append(box)
        labels.extend([1])

    space_time += truncnorm.rvs(-10, 10, loc=noise['mean'], scale=noise['std'], size=(len(t_vect), len(x_vect)))
    target = {
        'boxes': np.array(boxes),
        'labels': np.array(labels, dtype=np.int64),
    }
    return (space_time-np.mean(space_time))/np.std(space_time), target

# 数据集
class CustomDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]
        image_tensor = torch.tensor(image, dtype=torch.float).unsqueeze(0).repeat(3,1,1)
        boxes_tensor = torch.tensor(target['boxes'], dtype=torch.float32)
        labels_tensor = torch.tensor(target['labels'], dtype=torch.int64)
        target_tensor = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
        }

        return image_tensor, target_tensor

def collate_fn(batch):
    return tuple(zip(*batch))

class WaveLabel:
    def __init__(self, data, load_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data

