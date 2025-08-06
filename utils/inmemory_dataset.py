import numpy as np
from torch.utils.data import Dataset
from multiprocessing import Pool
from torch.utils.data import DataLoader

__all__ = ['InMemoryDataset']

class InMemoryDataset(Dataset):
    def __init__(self, data_list=None):
        super(InMemoryDataset, self).__init__()
        self.data_list = data_list or []

    def __getitem__(self, index):
        if isinstance(index, slice):
            return InMemoryDataset(data_list=self.data_list[index])
        elif isinstance(index, (int, np.integer)):
            return self.data_list[index]
        elif isinstance(index, list):
            return InMemoryDataset(data_list=[self.data_list[i] for i in index])
        else:
            raise TypeError(f'Invalid argument type: {type(index)} of {index}')

    def __len__(self):
        return len(self.data_list)

    def transform(self, transform_fn, num_workers=1, drop_none=False):
        with Pool(num_workers) as pool:
            data_list = pool.map(transform_fn, self.data_list)
        if drop_none:
            self.data_list = [data for data in data_list if data is not None]
        else:
            self.data_list = data_list

    def get_data_loader(self, batch_size, num_workers=1, shuffle=False, collate_fn=None):
        return DataLoader(self, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=shuffle,
            collate_fn=collate_fn)