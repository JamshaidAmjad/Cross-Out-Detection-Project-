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
import os
from data.transforms import get_train_transforms, get_val_transforms
from torchvision.datasets import ImageFolder
# Classes to include (MIXED is excluded)
from pathlib import Path
import shutil
import os
CLASS_NAMES = [
    "CLEAN",
    "CROSS",
    "DIAGONAL",
    "DOUBLE_LINE",
    "SCRATCH",
    "SINGLE_LINE",
    "WAVE",
    "ZIG_ZAG",
    "MIXED",
]

#CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}


def setup_data(dirpath:str):




    Multiclasses = [
    "CROSS",
    "DIAGONAL",
    "DOUBLE_LINE",
    "SCRATCH",
    "SINGLE_LINE",
    "WAVE",
    "ZIG_ZAG",
    ]
    binaryclasses=[
        "CLEAN",
        "MIXED"
    ]
    #make a multiclassfolder and a binaryfolder
    for split in ["train","test","val" ]:
        Path(f"{dirpath}/multiclass/{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{dirpath}/binaryclass/{split}/images").mkdir(parents=True, exist_ok=True)

    #copy validation
    for split2 in  ["train","test","val" ]:
        print(f"copying set {split2} ({['train','test','val'].index(split2)+1}/3)")
        for Class in Multiclasses:
            print(f"Multiclass: ({Multiclasses.index(Class)+1}/{len(Multiclasses)})")
            shutil.copytree(
            f"{dirpath}/{split2}/images/{Class}",
            f"{dirpath}/multiclass/{split2}/images/{Class}",
            dirs_exist_ok=True
        )
        for Class in binaryclasses:
            print(f"Binaryclass: ({binaryclasses.index(Class)+1}/{len(binaryclasses)})")
            shutil.copytree(
            f"{dirpath}/{split2}/images/{Class}",
            f"{dirpath}/binaryclass/{split2}/images/{Class}",
            dirs_exist_ok=True
        )
    
    #copy Training
    
    #copy test test






    
    
    



def Filter(data:ImageFolder,class_toremove:str):
    if isinstance(data,Subset):
        base=data.dataset
        indices=data.indices
    else:
        base=data
        indices=range(len(base))

    
    label_ignore=base.class_to_idx[class_toremove]
    
    new_indices=[]
    for i in indices:
        _,label=base.samples[i]
        if label != label_ignore:
            new_indices.append(i)

    return Subset(base, new_indices)









def get_dataloaders(
    
    multiclass:bool,
    Class_names:list=CLASS_NAMES,
    data_dir="data/ImageFolder",
    batch_size: int = 32,
    num_workers: int = 4,
    
    
    ):



    path= os.path.join(data_dir,"binaryclass")
    if not Path(path).is_dir():
        inp= input("Data Folders have to be formated in order to be used, do you wish to do this? (y/n)")
        if inp.lower()=="y":
            setup_data(dirpath=data_dir)
    
    

    #the path of the 
    if not multiclass:
        train_path = os.path.join(data_dir, "binaryclass", "train", "images")
        test_path  = os.path.join(data_dir, "binaryclass", "test", "images")
        val_path   = os.path.join(data_dir, "binaryclass", "val", "images")

    else:
        train_path = os.path.join(data_dir, "multiclass", "train", "images")
        test_path  = os.path.join(data_dir, "multiclass", "test", "images")
        val_path   = os.path.join(data_dir, "multiclass", "val", "images")



    #call the transform
    train_transform= get_train_transforms()
    val_transform=get_val_transforms()
    test_transform=get_val_transforms()#we use the same for both
    


    train_Dataset=ImageFolder(transform=train_transform,root=train_path)
    test_Dataset=ImageFolder(transform=test_transform,root=test_path)
    val_Dataset=ImageFolder(transform=val_transform,root=val_path)




   

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
    
