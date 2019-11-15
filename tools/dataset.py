import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, num_of_data_terms):
        self.term_list = range(num_of_data_terms)
        self.num_of_data_terms = num_of_data_terms

    def __len__(self):
        return self.num_of_data_terms

    def __getitem__(self, item):
        return self.term_list[item]