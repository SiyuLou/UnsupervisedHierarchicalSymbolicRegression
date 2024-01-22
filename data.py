import torch
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import h5py
from sklearn.model_selection import train_test_split,GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

class TLCDataset(Dataset):
    def __init__(self, dataframe, feature_names, feature_nums):
        """
        Arguments:
            dataframe (pd.DataFrame): Database in DataFrame form
            feature_names (list): Names of selected features
        """
        self.labels = dataframe['Rf'].values
        self.variables = dataframe.loc[:,feature_names].values
        
        feature_sequences = []
        feature_counts = 0
        self.feats = []
        for i, feature_num in enumerate(feature_nums):
            
            feature_sequences.append([feature_counts, feature_counts+feature_num])
            feature_counts += feature_num
            self.feats.append(self.variables[:,feature_sequences[i][0]:feature_sequences[i][1]])
 
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        final_feats = []
        for i, feat in enumerate(self.feats):
            feat = torch.tensor(feat[idx]).float()
            final_feats.append(feat)

        final_feats = tuple(final_feats)
        label = torch.tensor(self.labels[idx]).float().unsqueeze(dim=0)
        return final_feats, label

class SubmodelDataset(Dataset):
    def __init__(self, X_train, y_train, feature_nums):
        """
        Arguments:
            dataframe (pd.DataFrame): Database in DataFrame form
            feature_names (list): Names of selected features
        """
        self.labels = y_train
        self.variables = X_train
        
        feature_sequences = []
        feature_counts = 0
        self.feats = []
        for i, feature_num in enumerate(feature_nums):
            
            feature_sequences.append([feature_counts, feature_counts+feature_num])
            feature_counts += feature_num
            self.feats.append(self.variables[:,feature_sequences[i][0]:feature_sequences[i][1]])
 
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        final_feats = []
        for i, feat in enumerate(self.feats):
            feat = torch.tensor(feat[idx]).float()
            final_feats.append(feat)

        final_feats = tuple(final_feats)
        label = torch.tensor(self.labels[idx]).float().unsqueeze(dim=0)
        return final_feats, label


def get_data(feature_names, 
             feature_nums, 
             xlsx_file = '../../data/output_with_maccs.xlsx', 
             batch_size=1024, 
             random_state = 42,
             shuffle=True,
             savepath = './data'):
    
    df = pd.read_excel(xlsx_file, engine='openpyxl')
    ratio_train = 0.8
    ratio_val = 0.1
    ratio_test = 0.1

    # Produces test split.
    remaining, test= train_test_split(df, test_size=ratio_test, random_state=random_state)

    # Adjusts val ratio, w.r.t. remaining dataset.
    ratio_remaining = 1 - ratio_test
    ratio_val_adjusted = ratio_val / ratio_remaining

    # Produces train and val splits.
    train, val= train_test_split(remaining, test_size=ratio_val_adjusted, random_state=random_state)
    
    if savepath is not None:
        savepath = os.path.join(savepath, 'split_data')
        os.makedirs(savepath, exist_ok=True)
        train.to_csv(os.path.join(savepath, 'train.csv'), index=None)
        val.to_csv(os.path.join(savepath, 'dev.csv'), index=None)
        test.to_csv(os.path.join(savepath, 'test.csv'), index=None)

      
    train_dset = TLCDataset(train, feature_names, feature_nums)
    train_loader = DataLoader(dataset=train_dset, 
                              batch_size=batch_size, 
                              shuffle=shuffle)
    
    val_dset = TLCDataset(val, feature_names,feature_nums)
    val_loader = DataLoader(dataset=val_dset, 
                              batch_size=batch_size, 
                              shuffle=False)

    test_dset = TLCDataset(test, feature_names, feature_nums)
    test_loader  = DataLoader(dataset=test_dset,
                              batch_size=batch_size,
                              shuffle=False)
    print(f'the size of train dataset is {len(train_dset)}, validation_dataset is {len(val_dset)}, test dataset is {len(test_dset)}')
    return train_loader, val_loader, test_loader


def get_data_submodel(save_folder, feature_nums, shuffle=True):

    X_train = np.load(os.path.join(save_folder,'X_train.npy'))
    y_train = np.load(os.path.join(save_folder,'y_train.npy')).reshape(-1)
    X_val = np.load(os.path.join(save_folder,'X_val.npy'))
    y_val = np.load(os.path.join(save_folder,'y_val.npy')).reshape(-1)
    X_test = np.load(os.path.join(save_folder,'X_test.npy'))
    y_test = np.load(os.path.join(save_folder,'y_test.npy')).reshape(-1)
    
    train_dset = SubmodelDataset(X_train, y_train, feature_nums)
    train_loader = DataLoader(dataset=train_dset, 
                              batch_size=1024, 
                              shuffle=shuffle)
    val_dset = SubmodelDataset(X_val, y_val,feature_nums)
    val_loader = DataLoader(dataset=val_dset, 
                              batch_size=1024, 
                              shuffle=False)


    test_dset = SubmodelDataset(X_test, y_test, feature_nums)
    test_loader  = DataLoader(dataset=test_dset,
                              batch_size=1024,
                              shuffle=False)
    print(f'the size of train dataset is {len(train_dset)}, validation_dataset is {len(val_dset)}, test dataset is {len(test_dset)}')
    return train_loader, val_loader, test_loader
