"""
 This script defines the dataloader for the FML final project dataset.
"""
import json
import numpy as np
import pandas as pd
import torch.utils.data.dataset import Dataset

class FMLDataset(Dataset):
    def __init__(self, csv_path):
        """
        Argument: csv_path is the name like "./data/Amazon/Amazon.train.csv"
        """
        # read from csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # first column contains the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # second column contains the contexts
        self.context_arr = np.asarray(self.data_info.iloc[:, 1])
        # calculate length
        self.data_len = len(self.label_arr)
        
    def __getitem__(self, index):
        # get the context of the corresponding index
        context = self.context_arr[index]
        # get the label of the corresponding index
        label = self.label_arr[index]

        return (context, label)

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    # call the FML dataset
    fml_dataset = FMLDataset("../data/Amazon/amazon.train.csv")