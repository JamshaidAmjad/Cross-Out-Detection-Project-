"""
dataset.py
----------
PyTorch dataset loader for the Cross-Out Detection project.

Loads images using torchvision.datasets.ImageFolder, applies class filtering
to exclude the MIXED folder, and returns train/val/test DataLoaders via
get_dataloaders().

Class mapping:
    CLEAN=0, CROSS=1, DIAGONAL=2, DOUBLE_LINE=3,
    SCRATCH=4, SINGLE_LINE=5, WAVE=6, ZIG_ZAG=7
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset ,TensorDataset
from torchvision import datasets

from data.transforms import get_train_transforms, get_val_transforms
from torchvision.datasets import ImageFolder
# Classes to include (MIXED is excluded)


CLASS_NAMES = [
    "CLEAN",
    "CROSS",
    "DIAGONAL",
    "DOUBLE_LINE",
    "SCRATCH",
    "SINGLE_LINE",
    "WAVE",
    "ZIG_ZAG",
    
]

#CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

def Filter(data:ImageFolder,):
    label_ignore=data.class_to_idx["MIXED"]
    indices=[]
    for i ,(_,label) in enumerate(data.samples):
        if label != label_ignore:
            indices.append(i)

    return Subset(data, indices)









def get_dataloaders(
    
    
    Class_names=CLASS_NAMES,
    data_dir="data/ImageFolder/",
    batch_size: int = 32,
    num_workers: int = 4,
    
    ):


    
    

    #the path of the 
    train_path=data_dir+"train/Images"
    test_path=data_dir+"test/Images"
    val_path=data_dir+"val/Images"
    #call the transform
    train_transform= get_train_transforms()
    val_transform=get_val_transforms()
    test_transform=get_val_transforms()#we use the same for both
    


    train_Dataset=ImageFolder(transform=train_transform,root=train_path)
    test_Dataset=ImageFolder(transform=test_transform,root=test_path)
    val_Dataset=ImageFolder(transform=val_transform,root=val_path)




    #filter the MIXED class
    train_Dataset=Filter(train_Dataset)
    test_Dataset=Filter(test_Dataset)
    val_Dataset=Filter(val_Dataset)
    #create loaders-----------------------
    
    train_loader=DataLoader(batch_size=batch_size,shuffle=True,dataset=train_Dataset)
    val_loader=DataLoader(batch_size=batch_size,shuffle=False,dataset=val_Dataset)
    test_loader=DataLoader(batch_size=batch_size,shuffle=False,dataset=test_Dataset)

    return (train_loader, val_loader, test_loader)
    #dataloader.dataset.dataset.class_to_idx this is how you find the correct class
    
    # TODO: Load full dataset with ImageFolder using get_train_transforms()
    # TODO: Filter out the MIXED class using a custom is_valid_file or
    #       by overriding the class-to-index mapping
    # TODO: Split indices into train / val / test using torch.randperm or
    #       sklearn.model_selection.train_test_split with the given seed
    # TODO: Create Subset datasets for each split
    # TODO: Apply get_val_transforms() to val and test subsets
    # TODO: Wrap each Subset in a DataLoader with appropriate shuffle settings
    #       (shuffle=True for train, False for val/test)
    
