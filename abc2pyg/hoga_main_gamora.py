import argparse

import torch
import torch.utils.data as Data
import torch_geometric.transforms as T

from logger import Logger
import os
import sys
import numpy as np
import pandas as pd
import copy

from hoga_utils import *
from dataset_prep import PygNodePropPredDataset, Evaluator
from hoga_model import HOGA


#torch.set_num_threads(80)
torch.manual_seed(0)

def main():
    parser = argparse.ArgumentParser(description='mult16')
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--bits_test', type=int, default=32)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_hops', type=int, default=8)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--mapped', type=int, default=0)
    parser.add_argument('--lda1', type=int, default=5)
    parser.add_argument('--lda2', type=int, default=1)
    parser.add_argument('--design', type=str, default='mult')
    parser.add_argument('--root_dir', type=str, default='/scratch-x3/circuit_datasets')
    parser.add_argument('--directed', action='store_true')
    parser.add_argument('--test_all_bits', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--datagen', type=int, default=0,
        help="0=multiplier generator, 1=adder generator, 2=loading design")
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device = torch.device('cpu') ## cpu for now only

    if not os.path.exists(f'models/'):
        os.makedirs(f'models/')

    ## generate dataset
    root_folder = os.path.abspath((os.path.dirname(os.path.abspath("gnn_multitask.py"))))
    print(root_folder)
    # new generator functionalities: 0) multiplier 1) adder 2) read design

    if args.datagen == 0:
        prefix = 'mult'
    elif args.datagen == 1:
        prefix = 'adder'
    if args.mapped == 1:
        suffix ="_7nm_mapped"
    else:
        suffix = ''
    
    design_name = prefix + str(args.bits) + suffix
    
    ### training dataset loading
    dataset_r = PygNodePropPredDataset(name = design_name + '_root')
    print("Training on %s" % design_name)
    data_r = dataset_r[0]
    #data_r = T.ToSparseTensor()(data_r)
    
    dataset = PygNodePropPredDataset(name = design_name + '_shared')
    data = dataset[0]
    #data = T.ToSparseTensor()(data)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    ### training dataset loading
    #master = pd.read_csv('dataset_prep/master.csv', index_col = 0)
    #if not design_name_root in master:
    #    os.system(f"python dataset_prep/make_master_file.py --design_name {design_name_root}")
    #if not design_name_shared in master:
    #    os.system(f"python dataset_prep/make_master_file.py --design_name {design_name_shared}")
    dataset_r = PygNodePropPredDataset(name= design_name + '_root')#, root=root_path)
    print("Training on %s" % design_name)
    data_r = dataset_r[0]
    data_r = T.ToSparseTensor()(data_r)

    dataset = PygNodePropPredDataset(name=design_name + '_shared')#, root=root_path)
    data = dataset[0]
    data = preprocess(data, args)
    data = T.ToSparseTensor()(data)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']#.to(device)
    valid_idx = split_idx['valid']#.to(device)
    test_idx = split_idx['test']#.to(device)

    batch_data_train = Data.TensorDataset(data.x[train_idx], data.y[train_idx], data_r.y[train_idx])
    # batch_data_valid = Data.TensorDataset(data.x[valid_idx], data.y[valid_idx], data_r.y[valid_idx])
    batch_data_test = Data.TensorDataset(data.x[test_idx], data.y[test_idx], data_r.y[test_idx])

    train_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle=True, num_workers=10)
    # valid_loader = Data.DataLoader(batch_data_valid, batch_size=args.batch_size, shuffle=False, num_workers=10)
    test_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle=False, num_workers=10)

    model = HOGA(data.num_features, args.hidden_channels, 3, args.num_layers,
            args.dropout, num_hops=args.num_hops+1, heads=args.heads).to(device)

    logger_r = Logger(args.runs, args)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
        best_test_r = float('-inf')
        best_test_s = float('-inf')
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, train_loader, optimizer, device, args)
            result = test_all(model, test_loader, device)
            logger_r.add_result(run, result[:3])
            logger.add_result(run, result[3:])

            if epoch % args.log_steps == 0:
                train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s = result
                if test_acc_s >= best_test_s:
                    best_test_r = test_acc_r
                    best_test_s = test_acc_s
                    if args.save_model:
                        model_name = f'models/hoga_{design_name}_{args.design}.pt'
                        torch.save({'model_state_dict': model.state_dict()}, model_name)
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'[Root Model] Train: {100 * train_acc_r:.2f}%, '
                      f'[Root Model] Valid: {100 * valid_acc_r:.2f}% '
                      f'[Root Model] Test: {100 * test_acc_r:.2f}% '
                      f'[Shared Model] Train: {100 * train_acc_s:.2f}%, '
                      f'[Shared Model] Valid: {100 * valid_acc_s:.2f}% '
                      f'[Shared Model] Test: {100 * test_acc_s:.2f}%')

        logger_r.print_statistics(run)
        logger.print_statistics(run)
    logger_r.print_statistics()
    logger.print_statistics()

    ### evaluation dataset loading
    logger_eval_r = Logger(1, args)
    logger_eval = Logger(1, args)

    if args.mapped == 1:
        suffix ="_7nm_mapped"
    elif args.mapped == 2:
        suffix ="_mapped"
    else:
        suffix = ''

    if args.test_all_bits:
        bits_test_lst = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768]
    else:
        bits_test_lst = [args.bits_test]

    ## load pre-trained model
    if args.save_model:
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])

    for bits_test in bits_test_lst:
        if args.design == "booth":
            design_name = "booth_mult" + str(bits_test) + suffix
        else:
            design_name = "mult" + str(bits_test) + suffix
        design_name_root = design_name + "_root"
        design_name_shared = design_name + "_shared"
        print("Evaluation on %s" % design_name)

        #master = pd.read_csv('dataset_prep/master.csv', index_col = 0)
        #if not design_name_root in master:
        #    os.system(f"python dataset_prep/make_master_file.py --design_name {design_name_root}")
        #if not design_name_shared in master:
        #    os.system(f"python dataset_prep/make_master_file.py --design_name {design_name_shared}")
        #dataset_r = PygNodePropPredDataset(name=f'{design_name_root}', root=root_path)
        dataset_r = PygNodePropPredDataset(name= design_name + '_root')#, root=root_path)
        data_r = dataset_r[0]
        data_r = T.ToSparseTensor()(data_r)

        #dataset = PygNodePropPredDataset(name=f'{design_name_shared}', root=root_path)
        dataset = PygNodePropPredDataset(name= design_name + '_shared')#, root=root_path)
        data = dataset[0]
        data = preprocess(data, args)
        data = T.ToSparseTensor()(data)

        batch_data_test = Data.TensorDataset(data.x, data.y, data_r.y)
        test_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle=False, num_workers=10)

        for run_1 in range(1):
            for epoch in range(1):
                file_name = f'{args.design}_{design_name_shared}'
                result = test_all(model, test_loader, device, file_name)
                logger_eval_r.add_result(run_1, result[:3])
                logger_eval.add_result(run_1, result[3:])
                if epoch % args.log_steps == 0:
                    train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s = result
                    print(f'Run: {run_1 + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'[Root Model] Train: {100 * train_acc_r:.2f}%, '
                          f'[Root Model] Valid: {100 * valid_acc_r:.2f}% '
                          f'[Root Model] Test: {100 * test_acc_r:.2f}% '
                          f'[Shared Model] Train: {100 * train_acc_s:.2f}%, '
                          f'[Shared Model] Valid: {100 * valid_acc_s:.2f}% '
                          f'[Shared Model] Test: {100 * test_acc_s:.2f}%')

        logger_eval_r.print_statistics()
        logger_eval.print_statistics()

        ## save results
        if not os.path.exists(f'results/hoga'):
            os.makedirs(f'results/hoga')
        filename = f'results/hoga/{args.design}.csv'
        print(f"Saving results to {filename}")
        with open(f"{filename}", 'a+') as write_obj:
            write_obj.write(
                f"{design_name} " + f"{args.weight_decay} " + f"{args.dropout} " + f"{args.lr} " + \
                f"{args.num_layers} " + f"{args.epochs} " + f"{args.hidden_channels} " + \
                f"test_acc_r: {100 * test_acc_r:.2f} " + f"test_acc_s: {100 * test_acc_s:.2f} \n")

if __name__ == "__main__":
    main()
