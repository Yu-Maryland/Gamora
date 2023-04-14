import argparse

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d

from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from dataset_prep import PygNodePropPredDataset, Evaluator, GenMultDataset, ABCGenDataset, make_master

from logger import Logger
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='mult16')
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--datagen', type=int, default=0,
		help="0=multiplier generator, 1=adder generator, 2=loading design")
    # (0)(1) require bits as inputs; (2) requires designfile as input
    parser.add_argument('--datagen_test', type=int, default=0,
		help="0=multiplier generator, 1=adder generator, 2=loading design")
    parser.add_argument('--designfile', '-f', type=str, default='')
    parser.add_argument('--design_copies', type=int, default=1)
    args = parser.parse_args()
    print(args)

    ## generate dataset
    root_folder = os.path.abspath((os.path.dirname(os.path.abspath("ABC_dataset_generation.py"))))
    print(root_folder)
    # new generator functionalities: 0) multiplier 1) adder 2) read design

    import time
    start_time = time.time()
    dataset_generator = ABCGenDataset(args.bits, args.datagen, root_folder, args.designfile, 1) # shared labels
    dataset_generator.generate_batch(dataset_path=root_folder+"/dataset", train_split=0.8, val_split=1, batch=args.design_copies)
    design_name = dataset_generator.design_name + '_' + dataset_generator.prefix
    print("the time is %s s." % (time.time() - start_time))


    dataset_generator_r = ABCGenDataset(args.bits, args.datagen, root_folder, args.designfile, 0) # root labels
    dataset_generator_r.generate_batch(dataset_path=root_folder+"/dataset", train_split=0.8, val_split=1, batch=args.design_copies)
    design_name_root = dataset_generator_r.design_name + '_' + dataset_generator_r.prefix
    
    make_master(design_name, 6, 0)
    make_master(design_name_root, 5, 0)
    print("make_master :%s" % dataset_generator.design_name)


if __name__ == "__main__":
    main()
