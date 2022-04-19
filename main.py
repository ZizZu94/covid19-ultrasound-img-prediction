from utils.arguments import parse_arguments
from utils.dataset import CustomDataSet
from utils.transforms import get_transforms

from models.resnet50 import CustomResNet50

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
import torchvision
from torchvision.utils import make_grid
from torch.utils.data import DataLoader


def show_batch(dl):
    for images, _, info in dl:
        fig, ax = plt.subplots(figsize=(32, 32))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:32], nrow=8).squeeze().permute(1, 2, 0).clamp(0,1))
        plt.savefig("./img/mygraph.png")
        break

def experiments(args):
    # load data
    train_data = pd.read_csv(os.path.join(args.dataset_root, 'train_frames.csv'))
    test_data = pd.read_csv(os.path.join(args.dataset_root, 'test_frames.csv'))
    
    # subset the dataset
    train_ds = CustomDataSet(args, train_data, transforms=get_transforms(args, 'train'))
    test_ds = CustomDataSet(args, test_data, transforms=get_transforms(args, 'test'))

    train_labels = train_data['frame_score'].unique()
    num_calss = len(train_labels)
    
    # data loader
    train_dloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True)
    test_dloader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)

    # create log directories

    #show_batch(train_dloader)

if __name__ == '__main__':
    args = parse_arguments()
    experiments(args)