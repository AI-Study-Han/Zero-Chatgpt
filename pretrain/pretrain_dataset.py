
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split

class PretrainDataset(Dataset):
    def __init__(self, data_path_lst, max_length=512, memmap=False, seed=42):
        super().__init__()
        
        self.max_length = max_length
        self.seed = seed
        
        if memmap:
            with open(data_path_lst[0], 'rb') as f:
                nbytes = f.seek(0, 2)
                flen = nbytes // np.dtype('int16').itemsize  # 使用 int16 数据类型
            self.data = np.memmap(data_path_lst[0], dtype=np.dtype('int16'), shape=(flen // max_length, max_length), mode='r')
        else:
            data_lst = []
            for data_path in data_path_lst:
                with open(data_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.int16)  # 使用 int16 数据类型
                    data_lst.append(data)
            data = np.concatenate(data_lst)
            data = data[:max_length * (len(data) // max_length)]
            self.data = data.reshape(-1, max_length)
        
        self.indices = np.arange(len(self.data))
        np.random.shuffle(self.indices)
        print("memmap:{} train data.shape:{}".format(memmap, self.data.shape))
        print("downloading finished.....")
        
    def __len__(self):
        return self.data.shape[0]
    
    def shuffle_indices(self):
        np.random.seed(self.seed)

    def __getitem__(self, index: int):
        index = self.indices[index]
        sample = self.data[index]
        X = np.array(sample).astype(np.int64)
        Y = np.array(sample).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y)
    