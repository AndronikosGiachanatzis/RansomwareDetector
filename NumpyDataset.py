import torch
from torch.utils.data import Dataset

class numpy_dataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)

    def __getitem__(self, index):
        x = self.data[index]

        return x

    def __len__(self):
        return len(self.data)