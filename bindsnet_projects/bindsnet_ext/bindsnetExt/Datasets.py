import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from typing import Optional, Dict
import torchvision
from bindsnet.encoding import Encoder, NullEncoder

from sklearn import preprocessing

from .Utils import *


class spikeDataset(Dataset):
    
    '''
    PyTorch wrapper for a dataset already encoded into spike form.
    '''

    def __init__(self, 
                 subset_x, 
                 subset_y,
                 transform=None, 
                 target_transform=None):
                 
        x = subset_x
        y = subset_y
        
        self.dataset = (torch.as_tensor(subset_x, dtype=torch.uint8), torch.as_tensor(subset_y))
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.dataset[0][idx].unsqueeze(-2).unsqueeze(-2)
        
        #print(img.shape)
        
        target = self.dataset[1][idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'encoded_image':img, 'label':target}



class rbmDataset(Dataset):
    
    '''
    PyTorch wrapper for a dataset with RBM frequency encoding.
    It is required to pass a fitted instance of brbmGenerator class!
    '''

    def __init__(self, 
                 subset_x, 
                 subset_y,
                 rbm_encoder,
                 fold_num = 0, 
                 n_splits = None, 
                 train = True, 
                 transform=None, 
                 target_transform=None):
                 
        x = subset_x
        y = subset_y
        
        
        self.rbm_encoder = rbm_encoder
        
        if n_splits is not None:
        
            skf = StratifiedKFold(n_splits = n_splits, random_state = 31, shuffle = True)

            part_idx = []

            for i in range(0, n_splits):
                for train_idx, test_idx in skf.split(x,y):
                    part_idx.append((train_idx, test_idx))

            self.train_idx, self.test_idx = part_idx[fold_num]

            if train:
                self.dataset = (x[self.train_idx], y[self.train_idx])
            else:
                self.dataset = (x[self.test_idx], y[self.test_idx])
                
        else:
            self.dataset = (subset_x, subset_y)
            
        x_spikes = self.rbm_encoder.generate(self.dataset[0])
        
        self.dataset = (torch.as_tensor(x_spikes, dtype=torch.uint8), torch.as_tensor(subset_y))
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.dataset[0][idx].unsqueeze(-2).unsqueeze(-2)
        
        #print(img.shape)
        
        target = self.dataset[1][idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'encoded_image':img, 'label':target}


class picDataset(Dataset):
    
    '''
    PyTorch wrapper for any picture-like dataset
    '''

    def __init__(self, subset_x, subset_y, shape, random_state = None, fold_num = 0, n_splits = None, train = True, transform=None, target_transform=None):
                 
        x = subset_x
        y = subset_y
        self.shape = shape
        
        if n_splits is not None:
            
            if random_state is not None:
        
                skf = StratifiedKFold(n_splits = n_splits, random_state = random_state, shuffle = True)
            
            else:
                skf = StratifiedKFold(n_splits = n_splits, shuffle = True)

            part_idx = []

            for i in range(0, n_splits):
                for train_idx, test_idx in skf.split(x,y):
                    part_idx.append((train_idx, test_idx))

            self.train_idx, self.test_idx = part_idx[fold_num]

            if train:
                self.dataset = (x[self.train_idx], y[self.train_idx])
            else:
                self.dataset = (x[self.test_idx], y[self.test_idx])
                
        else:
            self.dataset = (subset_x, subset_y)
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.dataset[0][idx].astype(np.float32)
        img = img.reshape(self.shape)
        target = self.dataset[1][idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class bnDataset(picDataset):
    
    '''
    BindsNET wrapper for PyTorch picDataset wrapper (so you could wrap data while you wrap data)
    '''
        
    def __init__(
        self,
        image_encoder: Optional[Encoder] = None,
        label_encoder: Optional[Encoder] = None,
        *args,
        **kwargs
    ):
       
        super().__init__(*args, **kwargs)

        self.args = args
        self.kwargs = kwargs

        # Allow the passthrough of None, but change to NullEncoder
        if image_encoder is None:
            image_encoder = NullEncoder()

        if label_encoder is None:
            label_encoder = NullEncoder()

        self.image_encoder = image_encoder
        self.label_encoder = label_encoder

    def __getitem__(self, ind: int) -> Dict[str, torch.Tensor]:
        # language=rst
       
        image, label = super().__getitem__(ind)

        output = {
            "image": image,
            "label": label,
            "encoded_image": self.image_encoder(image),
            "encoded_label": self.label_encoder(label),
        }

        return output

    def __len__(self):
        return super().__len__()

    
class rfcDataset(Dataset):
    
    '''
    PyTorch spike wrapper for sklearn Iris & Cancer dataset
    '''

    def __init__(self,
                 subset_x,
                 subset_y,
                 train=True, 
                 fold_num = 0,
                 n_splits = None,
                 sigma = 0.005, 
                 dt = 0.1,
                 T = 100.,
                 n_coding_neurons = 20,
                 random_state = None,
                 scale = 1.,
                 transform=None, 
                 target_transform=None):
        
        x = subset_x
        y = subset_y
        
        
        if n_splits is not None:
            
            if random_state is not None:
        
                skf = StratifiedKFold(n_splits = n_splits, random_state = random_state, shuffle = True)
            
            else:
                skf = StratifiedKFold(n_splits = n_splits, shuffle = True)

            part_idx = []

            for i in range(0, n_splits):
                for train_idx, test_idx in skf.split(x,y):
                    part_idx.append((train_idx, test_idx))

            self.train_idx, self.test_idx = part_idx[fold_num]

            if train:
                self.dataset = (x[self.train_idx], y[self.train_idx])
            else:
                self.dataset = (x[self.test_idx], y[self.test_idx])
                
        else:
            self.dataset = (subset_x, subset_y)
        
        self.test_labels = y[self.test_idx]
        
        X = preprocessing.minmax_scale(self.dataset[0])
        Y = self.dataset[1].astype(np.int)

        round_to = round_decimals(dt)

        reverse = False
        no_last = False
        
        converter = ReceptiveFieldsConverter(sigma, 1.0, n_coding_neurons, round_to,
                                         scale=scale, reverse=reverse, no_last=no_last)

        sp = converter.convert(X,Y)
        
        #print(sp['input'])

        n_classes = len(np.unique(Y))

        samples, targets = getSpikes(sp, n_classes, Y, dt = dt, T = T)
        
        #print(samples.shape)
        #print(targets.shape)
                
        
        #if train:
        #    X, Y = samples[self.train_idx, :, :], targets[self.train_idx, :, :]
        #    
        #else:
        #    X, Y = samples[self.test_idx, :, :], targets[self.test_idx, :, :]
       
        self.dataset = (samples, targets)
        
        self.transform = transform
        self.target_transform = target_transform
        
    

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.dataset[0][idx,:, :]
        target = self.dataset[1][idx,:, :]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'X':img, 'Y':target}