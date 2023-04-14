import numpy as np
import networkx as nx
import os
import sys
import subprocess
from subprocess import check_output
import argparse
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath("gen_mult_dataset.py"))))
sys.path.append(root_folder)

import pandas as pd
import numpy as np
import random

import networkx as nx
import json
import os.path as osp
import shutil
from utils.torch_util import all_numpy
from datetime import date
from .dataset_pyg import PygNodePropPredDataset
from .dataset_pyg import make_master

class DatasetSaver(object):
    def __init__(self, dataset_name, root = '', version = 1):
        self.dataset_name = dataset_name
        self.root = root

        self.dataset_dir = osp.join(self.root, self.dataset_name)

        if osp.exists(self.dataset_dir):
            if input(f'Found an existing directory at {self.dataset_dir}/. \nWill you remove it? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.dataset_dir)
                print('Removed existing directory')
            else:
                print('Process stopped.')
                exit(-1)

        # make necessary dirs
        self.raw_dir = osp.join(self.dataset_dir, 'raw')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(osp.join(self.dataset_dir, 'processed'), exist_ok=True)

        # create release note
        with open(osp.join(self.dataset_dir, f'RELEASE_v{version}.txt'), 'w') as fw:
            fw.write(f'# Release note for {self.dataset_name}\n\n### v{version}: {date.today()}')

        # check list
        self._save_graph_list_done = False
        self._save_split_done = False

    def save_graph_list(self, graph_list):
        dict_keys = graph_list[0].keys()
        if not 'edge_index' in dict_keys:
            raise RuntimeError('edge_index needs to be provided in graph objects')
        if not 'num_nodes' in dict_keys:
            raise RuntimeError('num_nodes needs to be provided in graph objects')

        print(dict_keys)

        data_dict = {}
        # Store the following keys
        # - edge_index (necessary)
        # - num_nodes_list (necessary)
        # - num_edges_list (necessary)
        # - node_** (optional, node_feat is the default node features)
        # - edge_** (optional, edge_feat is the default edge features)

        # saving num_nodes_list
        print('Saving num-node-list.csv.gz')
        num_nodes_list = np.array([graph['num_nodes'] for graph in graph_list]).astype(np.int64)
        data_dict['num_nodes_list'] = num_nodes_list

        print('Saving edge.csv.gz and num-edge-list.csv.gz')
        edge_index = np.concatenate([graph['edge_index'] for graph in graph_list], axis = 1).astype(np.int64)
        num_edges_list = np.array([graph['edge_index'].shape[1] for graph in graph_list]).astype(np.int64)
        if edge_index.shape[0] != 2:
            raise RuntimeError('edge_index must have shape (2, num_edges)')
        data_dict['edge_index'] = edge_index
        data_dict['num_edges_list'] = num_edges_list

        for key in dict_keys:
            if key == 'edge_index' or key == 'num_nodes':
                continue
            if graph_list[0][key] is None:
                continue

            if 'node_' == key[:5]:
                # make sure saved in np.int64 or np.float32
                dtype = np.int64 if 'int' in str(graph_list[0][key].dtype) else np.float32
                # check num_nodes
                for i in range(len(graph_list)):
                    if len(graph_list[i][key]) != num_nodes_list[i]:
                        raise RuntimeError(f'num_nodes mistmatches with {key}')

                cat_feat = np.concatenate([graph[key] for graph in graph_list], axis = 0).astype(dtype)
                data_dict[key] = cat_feat

            elif 'edge_' == key[:5]:
                # make sure saved in np.int64 or np.float32
                dtype = np.int64 if 'int' in str(graph_list[0][key].dtype) else np.float32
                # check num_edges
                for i in range(len(graph_list)):
                    if len(graph_list[i][key]) != num_edges_list[i]:
                        raise RuntimeError(f'num_edges mistmatches with {key}')

                cat_feat = np.concatenate([graph[key] for graph in graph_list], axis = 0).astype(dtype)
                data_dict[key] = cat_feat

            else:
                raise RuntimeError(f'Keys in graph object should start from either \'node_\' or \'edge_\', but \'{key}\' given.')

        self.has_node_attr = ('node_feat' in graph_list[0]) and (graph_list[0]['node_feat'] is not None)
        self.has_edge_attr = ('edge_feat' in graph_list[0]) and (graph_list[0]['edge_feat'] is not None)

        # num-node-list, num-edge-list
        n_node_list = pd.DataFrame(data_dict['num_nodes_list'])
        n_edge_list = pd.DataFrame(data_dict['num_edges_list'])

        n_node_list.to_csv(self.raw_dir + '/num-node-list.csv', index = False, header = False)
        n_node_list.to_csv(self.raw_dir + '/num-node-list.csv.gz', index = False, header = False, compression='gzip')
        n_edge_list.to_csv(self.raw_dir + '/num-edge-list.csv', index = False, header = False)
        n_edge_list.to_csv(self.raw_dir + '/num-edge-list.csv.gz', index = False, header = False, compression='gzip')

        # edge list
        EDGE_list = pd.DataFrame(data_dict['edge_index'].transpose())
        EDGE_list.to_csv(self.raw_dir + '/edge.csv', index = False, header = False)
        EDGE_list.to_csv(self.raw_dir + '/edge.csv.gz', index = False, header = False, compression='gzip')

        # node-feat
        if self.has_node_attr:
            print('Saving node-feat.csv.gz')
            NODE = pd.DataFrame(data_dict['node_feat'])
            NODE.to_csv(self.raw_dir + '/node-feat.csv', index = False, header = False)
            NODE.to_csv(self.raw_dir + '/node-feat.csv.gz', index = False, header = False, compression='gzip')

        if self.has_edge_attr:
            print('Saving edge-feat.csv.gz')
            EDGE_feat = pd.DataFrame(data_dict['edge_feat'])
            EDGE_feat.to_csv(self.raw_dir + '/edge-feat.csv', index = False, header = False)
            EDGE_feat.to_csv(self.raw_dir + '/edge-feat.csv.gz', index = False, header = False, compression='gzip')

        print('Saved all the files!')
        self._save_graph_list_done = True
        self.num_data = graph_list[0]['num_nodes']

    def save_target_labels(self, target_labels):
        '''
            target_label (numpy.narray): storing target labels. Shape must be (num_data, num_tasks)
        '''

        if not self._save_graph_list_done:
            raise RuntimeError('save_graph_list must be done beforehand.')

        # check type and shape
        if not isinstance(target_labels, np.ndarray):
            raise ValueError(f'target label must be of type np.ndarray')

        if len(target_labels) != self.num_data:
            raise RuntimeError(f'The length of target_labels ({len(target_labels)}) must be the same as the number of data points ({self.num_data}).')

        node_label = pd.DataFrame(target_labels)
        node_label.to_csv(self.raw_dir + '/node-label.csv.gz', index = False, header = False, compression='gzip')
        node_label.to_csv(self.raw_dir + '/node-label.csv', index = False, header = False)

        self.num_tasks = target_labels.shape[1]

        self._save_target_labels_done = True


    def save_split(self, split_dict, split_name = 'random'):
        '''
            Save dataset split
                split_dict: must contain three keys: 'train', 'valid', 'test', where the values are the split indices stored in numpy.
                split_name (str): the name of the split
        '''

        self.split_dir = osp.join(self.dataset_dir, 'split', split_name)
        os.makedirs(self.split_dir, exist_ok = True)

        # verify input
        if not 'train' in split_dict:
            raise ValueError('\'train\' needs to be given in save_split')
        if not 'valid' in split_dict:
            raise ValueError('\'valid\' needs to be given in save_split')
        if not 'test' in split_dict:
            raise ValueError('\'test\' needs to be given in save_split')

        if not all_numpy(split_dict):
            raise RuntimeError('split_dict must only contain list/dict of numpy arrays, int, or float')

        test_list = pd.DataFrame(split_dict['test'])
        train_list = pd.DataFrame(split_dict['train'])
        valid_list = pd.DataFrame(split_dict['valid'])

        test_list.to_csv(self.split_dir + '/test.csv', index = False, header = False)
        train_list.to_csv(self.split_dir + '/train.csv', index = False, header = False)
        valid_list.to_csv(self.split_dir + '/valid.csv', index = False, header = False)

        test_list.to_csv(self.split_dir + '/test.csv.gz', index = False, header = False, compression='gzip')
        train_list.to_csv(self.split_dir + '/train.csv.gz', index = False, header = False, compression='gzip')
        valid_list.to_csv(self.split_dir + '/valid.csv.gz', index = False, header = False, compression='gzip')

        self.split_name = split_name
        self._save_split_done = True


# multiplier dataset generator
# input is number of bits (INT), train and val split ratio
class GenMultDataset(object):
    def __init__(self, bits, root = ''):
        # check list
        self.bits = bits
        self.design_name = "mult" + str(bits)
        self.root = root
        self.design_root = self.design_name + '_raw/'
    def generate(self, dataset_path, train_split=0.8, val_split=0.9):
        self._genmult_abc(self.bits)
        design_root = self.design_root
        class_name = design_root + self.design_name + "-class_map.json"
        file_edge_list = design_root + self.design_name + ".el"
        file_node_feat = design_root + self.design_name + "-feats.csv"
        save_dir = dataset_path + "/" + self.design_name + "/"
        print("saving dataset %s to %s" % (self.design_name,save_dir))
        graph_list = []
        # build graphs
        fh = open(file_edge_list, "rb")
        g = nx.read_edgelist(fh, create_using = nx.DiGraph, nodetype=int)
        fh.close()
        graph = dict()
        graph['edge_index'] = np.array(g.edges).transpose()
        graph['num_nodes'] = len(g.nodes)
        feats = np.loadtxt(file_node_feat, delimiter=',')
        graph['node_feat'] = np.array(feats)

        graph_list.append(graph)

        # build dataset directory
        saver = DatasetSaver(save_dir)
        # save graph info
        saver.save_graph_list(graph_list)

        # node label
        # read node label
        f_class = open(class_name)
        v = json.load(f_class).values()
        f_class.close()
        size = len(list(v))
        labels = np.argmax(np.array(list(v)), axis = 1).reshape(size, 1)

        # save node labels
        saver.save_target_labels(labels)

        # save train, valid, and test
        split_idx = dict()
        perm = np.random.permutation(saver.num_data)
        split_idx['train'] = perm[: int(0.8 * saver.num_data)]
        split_idx['valid'] = perm[int(0.8 * saver.num_data): int(0.9 * saver.num_data)]
        split_idx['test'] = perm[int(0.9 * saver.num_data):]
        saver.save_split(split_idx, split_name = 'Random')
    def _genmult_abc(self, bits):
        os.makedirs(self.design_root, exist_ok=True)
        cmd = "./abc -c \"gen -N %d -m mult%d.blif;strash;&get;&get; &edgelist -F %smult%d.el -c %smult%d-class_map.json -f %smult%d-feats.csv\"" % (self.bits, self.bits, self.design_root, self.bits, self.design_root, self.bits, self.design_root, self.bits)
        os.system(cmd)


# generic ABC dataset generator
class ABCGenDataset(object):
    def __init__(self, bits = 8, gentype = 0, root = '', designfile = '', multilabel = 0):
        # multilabel = 0
        # multilabel = 1 : increased number of classes (6)
        # multilabel = 2 : multihot representation
        # check list
        self.bits = bits
        self.gentype = gentype
        self.root = root
        self.designfile = designfile
        self.multilabel = multilabel
        if self.gentype==0: # mult
            if multilabel == 0:
                self.prefix = 'root'
            elif multilabel == 1:
                self.prefix = 'shared'
            else:
                self.prefix = 'mhot'
            self.design_name = "mult" + str(bits) 
            self.design_root = self.design_name + '_' + self.prefix + '_raw/'
            #self.design_root = self.design_name + '_raw/'
            self._genmult_abc(self.bits)
        elif self.gentype==1: # add
            if multilabel == 0:
                self.prefix = 'root'
            elif multilabel == 1:
                self.prefix = 'shared'
            else:
                self.prefix = 'mhot'
            self.design_name = "adder" + str(bits)
            self.design_root = self.design_name + '_' + self.prefix + '_raw/'
            self._genadd_abc(self.bits)
        elif self.gentype==2: # read design
            assert(self.designfile != ''), "****Loading design for data generation needs design filename as input****"
            self.design_name = self.designfile
            self.design_root = self.design_name + '_raw/'
            self._read_abc(self.bits)

    def generate(self, dataset_path, train_split=0.8, val_split=0.9):
        design_root = self.design_root
        class_name = design_root + self.design_name + "-class_map.json"
        file_edge_list = design_root + self.design_name + ".el"
        file_node_feat = design_root + self.design_name + "-feats.csv"
        # save_dir = dataset_path + "/" + self.design_name + '_' + str(self.multilabel) + "/"
        # save_dir = dataset_path + "/" + self.design_name + '_' + self.prefix + "/"
        save_dir = dataset_path + "/" + self.design_name + "/"
        print("saving dataset %s to %s" % (self.design_name, save_dir))
        graph_list = []
        # build graphs
        fh = open(file_edge_list, "rb")
        g = nx.read_edgelist(fh, create_using = nx.DiGraph, nodetype=int)
        fh.close()
        graph = dict()
        graph['edge_index'] = np.array(g.edges).transpose()
        graph['num_nodes'] = len(g.nodes)
        feats = np.loadtxt(file_node_feat, delimiter=',')
        graph['node_feat'] = np.array(feats)

        graph_list.append(graph)

        # build dataset directory
        saver = DatasetSaver(save_dir)
        # save graph info
        saver.save_graph_list(graph_list)

        # node label
        # read node label
        f_class = open(class_name)
        v = json.load(f_class).values()
        f_class.close()
        size = len(list(v))
        if self.multilabel == 2:
            labels = np.array(list(v)).reshape(size, 5)
            print(labels[0])
        elif self.multilabel == 1:
            # labels = [0 for i in range(size)]
            # v = list(v)
            # for i in range(size):
            #     l = np.argmax(v[i])
            #     if l == 2:
            #         if v[i][7]: # shared + maj root
            #             l = 6
            #         elif v[i][6]: # shared + xor root
            #             l = 7
            #     labels[i] = l
            # labels = np.array(labels).reshape(size, 1)
            labels = np.argmax(np.array(list(v)), axis = 1).reshape(size, 1)
        else:
            labels = np.argmax(np.array(list(v)), axis = 1).reshape(size, 1)

        # save node labels
        saver.save_target_labels(labels)

        # save train, valid, and test
        split_idx = dict()
        perm = np.random.permutation(saver.num_data)
        split_idx['train'] = perm[: int(train_split * saver.num_data)]
        split_idx['valid'] = perm[int(train_split * saver.num_data): int(val_split * saver.num_data)]
        split_idx['test'] = perm[int((1-val_split) * saver.num_data):]
        saver.save_split(split_idx, split_name = 'Random')
        make_master(self.design_name, 5 + self.multilabel, 1)
        
    # repeat one design for multiple times
    def generate_batch(self, dataset_path, train_split=0.8, val_split=0.9, batch=1):
        design_root = self.design_root
        if self.gentype == 0:
            class_name = design_root + self.design_name + "-class_map.csv"
        else:
            class_name = design_root + self.design_name + "-class_map.json"
        
        file_edge_list = design_root + self.design_name + ".el"
        file_node_feat = design_root + self.design_name + "-feats.csv"
        # save_dir = dataset_path + "/" + self.design_name + '_' + str(self.multilabel) + "/"
        # save_dir = dataset_path + "/" + self.design_name + "/"
        if batch > 1: 
            save_dir = dataset_path + "/" + self.design_name + '_' + self.prefix + '_batch_' + str(batch) + "/"
        else:
            save_dir = dataset_path + "/" + self.design_name + '_' + self.prefix + "/"
        print("saving dataset %s to %s" % (self.design_name,save_dir))
        graph_list = []
        # build graphs
        fh = open(file_edge_list, "rb")
        g = nx.read_edgelist(fh, create_using = nx.DiGraph, nodetype=int)
        fh.close()
        graph = dict()
        num_nodes = len(g.nodes)
        graph['edge_index'] = np.array(g.edges).transpose()
        feats = np.loadtxt(file_node_feat, delimiter=',')
        graph['node_feat'] = np.array(feats)
        
        for i in range(batch - 1):
            new_edges = np.add(np.array(g.edges).transpose(), (i + 1) * num_nodes)
            # h = g.copy()
            # for n in h.nodes:
            #     mapping = {n: int(n) + (i + 1) * num_nodes}
            #     h = nx.relabel_nodes(h, mapping)
            # graph['edge_index'] = np.concatenate((graph['edge_index'], np.array(h.edges).transpose()), axis = 1)
            graph['edge_index'] = np.concatenate((graph['edge_index'], new_edges), axis = 1)
            graph['node_feat'] = np.concatenate((graph['node_feat'], np.array(feats)), axis = 0)

        graph['num_nodes'] = int(batch * len(g.nodes))
        
        graph_list.append(graph)

        # build dataset directory
        saver = DatasetSaver(save_dir)
        # save graph info
        saver.save_graph_list(graph_list)

        # node label
        # read node label
        
        # csv version
        v = np.loadtxt(class_name, delimiter=',')
        
        # f_class = open(class_name)
        # v = json.load(f_class).values()
        # f_class.close()
        size = len(list(v))
        if self.multilabel == 2:
            labels = np.array(list(v)).reshape(size, 5)
            # print(labels[0])
        elif self.multilabel == 1:
            labels = np.argmax(np.array(list(v)), axis = 1).reshape(size, 1)
        else:
            labels = np.argmax(np.array(list(v)), axis = 1).reshape(size, 1)
        
        for i in range(batch - 1):
            labels = np.concatenate((labels,  np.argmax(np.array(list(v)), axis = 1).reshape(size, 1)), axis = 0)

        # save node labels
        saver.save_target_labels(labels)

        # save train, valid, and test
        split_idx = dict()
        perm = np.random.permutation(saver.num_data)
        split_idx['train'] = perm[: int(train_split * saver.num_data)]
        split_idx['valid'] = perm[int(train_split * saver.num_data): int(val_split * saver.num_data)]
        split_idx['test'] = perm[int((1-val_split) * saver.num_data):]
        saver.save_split(split_idx, split_name = 'Random')
        
        
    def _genmult_abc(self, bits, booth=False): # booth multiplier not implemented yet 05/30/22
        os.makedirs(self.design_root, exist_ok=True)
        # if self.multilabel == 1:
        #     cmd = "./abc -c \"gen -N %d -m mult%d.blif;strash;&get;&get; &edgelist -F %smult%d.el -c %smult%d-class_map.json -f %smult%d-feats.csv -m\"" % (self.bits, self.bits, self.design_root, self.bits, self.design_root, self.bits, self.design_root, self.bits)
        # elif self.multilabel == 2:
        #     cmd = "./abc -c \"gen -N %d -m mult%d.blif;strash;&get;&get; &edgelist -F %smult%d.el -c %smult%d-class_map.json -f %smult%d-feats.csv -M\"" % (self.bits, self.bits, self.design_root, self.bits, self.design_root, self.bits, self.design_root, self.bits)
        # else:
        #     cmd = "./abc -c \"gen -N %d -m mult%d.blif;strash;&get;&get; &edgelist -F %smult%d.el -c %smult%d-class_map.json -f %smult%d-feats.csv\"" % (self.bits, self.bits, self.design_root, self.bits, self.design_root, self.bits, self.design_root, self.bits)
        if self.multilabel == 1:
            cmd = "./abc -c \"gen -N %d -m mult%d.blif;strash;&get;&get; &edgelist -F %smult%d.el -c %smult%d-class_map.csv -f %smult%d-feats.csv -m\"" % (self.bits, self.bits, self.design_root, self.bits, self.design_root, self.bits, self.design_root, self.bits)
        elif self.multilabel == 2:
            cmd = "./abc -c \"gen -N %d -m mult%d.blif;strash;&get;&get; &edgelist -F %smult%d.el -c %smult%d-class_map.csv -f %smult%d-feats.csv -M\"" % (self.bits, self.bits, self.design_root, self.bits, self.design_root, self.bits, self.design_root, self.bits)
        else:
            cmd = "./abc -c \"gen -N %d -m mult%d.blif;strash;&get;&get; &edgelist -F %smult%d.el -c %smult%d-class_map.csv -f %smult%d-feats.csv\"" % (self.bits, self.bits, self.design_root, self.bits, self.design_root, self.bits, self.design_root, self.bits)
        os.system(cmd)
    def _genadd_abc(self, bits):
        os.makedirs(self.design_root, exist_ok=True)
        if self.multilabel == 1:
            cmd = "./abc -c \"gen -N %d -a adder%d.blif;strash;&get;&get; &edgelist -F %sadder%d.el -c %sadder%d-class_map.json -f %sadder%d-feats.csv -m\"" % (self.bits, self.bits, self.design_root, self.bits, self.design_root, self.bits, self.design_root, self.bits)
        elif self.multilabel == 2:
            cmd = "./abc -c \"gen -N %d -a adder%d.blif;strash;&get;&get; &edgelist -F %sadder%d.el -c %sadder%d-class_map.json -f %sadder%d-feats.csv -M\"" % (self.bits, self.bits, self.design_root, self.bits, self.design_root, self.bits, self.design_root, self.bits)
        else:
            cmd = "./abc -c \"gen -N %d -a adder%d.blif;strash;&get;&get; &edgelist -F %sadder%d.el -c %sadder%d-class_map.json -f %sadder%d-feats.csv\"" % (self.bits, self.bits, self.design_root, self.bits, self.design_root, self.bits, self.design_root, self.bits)
        os.system(cmd)
    def _read_abc(self, bits):
        os.makedirs(self.design_root, exist_ok=True)
        if self.multilabel == 1:
            cmd = "./abc -c \"read %s;strash;&get;&get; &edgelist -F %s%s.el -c %s%s-class_map.json -f %s%s-feats.csv -m\"" % (self.designfile, self.design_root, self.designfile, self.design_root, self.designfile, self.design_root, self.designfile)
        elif self.multilabel == 2:
            cmd = "./abc -c \"read %s;strash;&get;&get; &edgelist -F %s%s.el -c %s%s-class_map.json -f %s%s-feats.csv -M\"" % (self.designfile, self.design_root, self.designfile, self.design_root, self.designfile, self.design_root, self.designfile)
        else:
            cmd = "./abc -c \"read %s;strash;&get;&get; &edgelist -F %s%s.el -c %s%s-class_map.json -f %s%s-feats.csv\"" % (self.designfile, self.design_root, self.designfile, self.design_root, self.designfile, self.design_root, self.designfile)
        os.system(cmd)

class GenDataset_batch(object):
    def __init__(self, bits = 8, gentype = 0, root = '', designfile = '', multilabel = 0):
        # multilabel = 0
        # multilabel = 1 : increased number of classes (6)
        # multilabel = 2 : multihot representation
        # check list
        self.bits = bits
        self.gentype = gentype
        self.root = root
        self.designfile = designfile
        self.multilabel = multilabel
        if self.gentype==0: # mult
            if multilabel == 0:
                self.prefix = 'root'
            elif multilabel == 1:
                self.prefix = 'shared'
            else:
                self.prefix = 'mhot'
            self.design_name = "mult" + str(bits) 
            self.design_root = self.design_name + '_' + self.prefix + '_raw/'
        elif self.gentype==1: # add
            self.design_name = "adder" + str(bits)
            self.design_root = self.design_name + '_raw/'
        elif self.gentype==2: # read design
            assert(self.designfile != ''), "****Loading design for data generation needs design filename as input****"
            self.design_name = self.designfile
            self.design_root = self.design_name + '_raw/'
            
    def generate_batch(self, dataset_path, train_split=0.8, val_split=0.9, batch=1):
        design_root = self.design_root
        # class_name = design_root + self.design_name + "-class_map.json"
        class_name = design_root + self.design_name + "-class_map.csv"
        file_edge_list = design_root + self.design_name + ".el"
        file_node_feat = design_root + self.design_name + "-feats.csv"
        # save_dir = dataset_path + "/" + self.design_name + '_' + str(self.multilabel) + "/"
        # save_dir = dataset_path + "/" + self.design_name + "/"
        if batch > 1: 
            save_dir = dataset_path + "/" + self.design_name + '_' + self.prefix + '_batch_' + str(batch) + "/"
        else:
            save_dir = dataset_path + "/" + self.design_name + '_' + self.prefix + "/"
        print("saving dataset %s to %s" % (self.design_name,save_dir))
        graph_list = []
        # build graphs
        fh = open(file_edge_list, "rb")
        g = nx.read_edgelist(fh, create_using = nx.DiGraph, nodetype=int)
        fh.close()
        graph = dict()
        num_nodes = len(g.nodes)
        graph['edge_index'] = np.array(g.edges).transpose()
        feats = np.loadtxt(file_node_feat, delimiter=',')
        graph['node_feat'] = np.array(feats)
        
        for i in range(batch - 1):
            new_edges = np.add(np.array(g.edges).transpose(), (i + 1) * num_nodes)
            # h = g.copy()
            # for n in h.nodes:
            #     mapping = {n: int(n) + (i + 1) * num_nodes}
            #     h = nx.relabel_nodes(h, mapping)
            # graph['edge_index'] = np.concatenate((graph['edge_index'], np.array(h.edges).transpose()), axis = 1)
            graph['edge_index'] = np.concatenate((graph['edge_index'], new_edges), axis = 1)
            graph['node_feat'] = np.concatenate((graph['node_feat'], np.array(feats)), axis = 0)

        graph['num_nodes'] = int(batch * len(g.nodes))
        
        graph_list.append(graph)

        # build dataset directory
        saver = DatasetSaver(save_dir)
        # save graph info
        saver.save_graph_list(graph_list)

        # node label
        # read node label
        
        # csv version
        v = np.loadtxt(class_name, delimiter=',')
        
        # f_class = open(class_name)
        # v = json.load(f_class).values()
        # f_class.close()
        size = len(list(v))
        if self.multilabel == 2:
            labels = np.array(list(v)).reshape(size, 5)
            # print(labels[0])
        elif self.multilabel == 1:
            labels = np.argmax(np.array(list(v)), axis = 1).reshape(size, 1)
        else:
            labels = np.argmax(np.array(list(v)), axis = 1).reshape(size, 1)
        
        for i in range(batch - 1):
            labels = np.concatenate((labels,  np.argmax(np.array(list(v)), axis = 1).reshape(size, 1)), axis = 0)

        # save node labels
        saver.save_target_labels(labels)

        # save train, valid, and test
        split_idx = dict()
        perm = np.random.permutation(saver.num_data)
        split_idx['train'] = perm[: int(train_split * saver.num_data)]
        split_idx['valid'] = perm[int(train_split * saver.num_data): int(val_split * saver.num_data)]
        split_idx['test'] = perm[int((1-val_split) * saver.num_data):]
        saver.save_split(split_idx, split_name = 'Random')
       




def main(args):
    generator = ABCGenDataset(bits = args.bits, gentype = args.gentype, root = root_folder, designfile = '', multilabel = 0)
    # generator.generate(dataset_path=root_folder+"/dataset")
    generator.generate_batch(dataset_path=root_folder+"/dataset", batch = 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, default=2)
    parser.add_argument('--gentype', type=int, default=0)
    parser.add_argument('--design-name', type=str, default='mult8')

    args_ = parser.parse_args()
    main(args_)
