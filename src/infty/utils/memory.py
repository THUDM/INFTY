import torch
import torch.utils.data as data

class Storage(data.Dataset):
    """
    A dataset wrapper used as a memory to store the data
    """
    def __init__(self):
        super(Storage, self).__init__()
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, index):
        return self.storage[index]

    def append(self,x):
        self.storage.append(x)

    def extend(self,x):
        self.storage.extend(x)

class Memory(Storage):
    def reduce(self, m):
        self.storage = self.storage[:m]

    def get_tensor(self):
        storage = [x.unsqueeze(-1) if isinstance(x, torch.Tensor) else torch.tensor(x).unsqueeze(-1) for x in self.storage]
        return torch.cat(storage, dim=1)