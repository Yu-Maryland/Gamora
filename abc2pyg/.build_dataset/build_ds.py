#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import networkx as nx
import os
import sys
import subprocess
from subprocess import check_output

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath("build_ds.ipynb"))))
sys.path.append(root_folder)


# In[58]:


import pandas as pd
import numpy as np
import random

import networkx as nx
import json
import os.path as osp
import shutil
from utils.torch_util import all_numpy
from datetime import date


# ## DatasetSaver for node-level classification dataset

# In[47]:


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


# In[48]:


class_name = "mult8_raw/mult8-class_map.json"
file_edge_list = "mult8_raw/mult8.el"
file_node_feat = "mult8_raw/mult8-feats.csv"
save_dir = "mult8/"
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


# In[49]:


# build dataset directory
saver = DatasetSaver('test')


# In[50]:


# save graph info
saver.save_graph_list(graph_list)


# In[51]:


### node label
# read node label
f_class = open(class_name)
v = json.load(f_class).values()
f_class.close()
size = len(list(v))
labels = np.argmax(np.array(list(v)), axis = 1).reshape(size, 1)

# save node labels
saver.save_target_labels(labels)


# In[54]:


# save train, valid, and test
split_idx = dict()
perm = np.random.permutation(saver.num_data)
split_idx['train'] = perm[: int(0.8 * saver.num_data)]
split_idx['valid'] = perm[int(0.8 * saver.num_data): int(0.9 * saver.num_data)]
split_idx['test'] = perm[int(0.9 * saver.num_data):]
saver.save_split(split_idx, split_name = 'random')


# ## Manually building dataset

# In[16]:


class_name = "mult8_raw/mult8-class_map.json"
file_edge_list = "mult8_raw/mult8.el"
file_node_feat = "mult8_raw/mult8-feats.csv"
save_dir = "mult8/"


# In[30]:


# read node labels
f_class = open(class_name)
v = json.load(f_class).values()
f_class.close()
size = len(list(v))
labels = np.argmax(np.array(list(v)), axis = 1).reshape(size, 1)

# save node labels
node_label = pd.DataFrame(labels.flatten())
node_label.to_csv(save_dir + 'node-label.csv.gz', index = False, header = False, compression='gzip')
node_label.to_csv(save_dir + 'node-label.csv', index = False, header = False)


# In[23]:


# build graphs
fh = open(file_edge_list, "rb")
g = nx.read_edgelist(fh, create_using = nx.DiGraph, nodetype=int)
fh.close()

feats = np.loadtxt(file_node_feat, delimiter=',')
d = dict(enumerate(feats.tolist(), 0))
nx.set_node_attributes(g, d, "feat")


num_node = [len(g.nodes)]
num_edge = [len(g.edges)]

node_feat=[]
edge_list=[]
edge_feat=[]

for i in range(len(g.nodes)):
    node_feat.append(list(g.nodes[i]['feat']))
    
for e in g.edges:
    source = e[0]
    target = e[1]
    edge_list.append([source, target])


# In[24]:


np.array(g.edges).transpose() 


# In[19]:


graph_list = []
num_data = 100
for i in range(num_data):
    g = nx.fast_gnp_random_graph(10, 0.5)
    graph = dict()
    graph['edge_index'] = np.array(g.edges).transpose() 
    num_edges = graph['edge_index'].shape[1]

    graph['num_nodes'] = len(g.nodes)
    # optionally, add node/edge features
    graph['node_feat'] = np.random.randn(graph['num_nodes'], 3)
    graph['edge_feat'] = np.random.randn(num_edges, 3) 
    
    graph_list.append(graph)


# In[20]:


print(graph_list[0].keys())


# In[29]:


# save node feats and edge lists
NODE = pd.DataFrame(node_feat)
EDGE_list = pd.DataFrame(edge_list)
#EDGE_feat=pd.DataFrame(edge_feat)

NODE.to_csv(save_dir + 'node-feat.csv', index = False, header = False)
NODE.to_csv(save_dir + 'node-feat.csv.gz', index = False, header = False, compression='gzip')
EDGE_list.to_csv(save_dir + 'edge.csv', index = False, header = False)
EDGE_list.to_csv(save_dir + 'edge.csv.gz', index = False, header = False, compression='gzip')
#EDGE_feat.to_csv('edge-feat.csv', index = False, header = False)


# In[31]:


# save num of nodes and num od edges
num_node_list = pd.DataFrame(num_node)
num_edge_list = pd.DataFrame(num_edge)

num_node_list.to_csv(save_dir + 'num-node-list.csv', index = False, header = False)
num_node_list.to_csv(save_dir + 'num-node-list.csv.gz', index = False, header = False, compression='gzip')
num_edge_list.to_csv(save_dir + 'num-edge-list.csv', index = False, header = False)
num_edge_list.to_csv(save_dir + 'num-edge-list.csv.gz', index = False, header = False, compression='gzip')


# In[38]:


from sklearn import model_selection
num = len(g.nodes)
train_valid, test= model_selection.train_test_split([i for i in range(num)], train_size=0.8, test_size = None)
train, valid = model_selection.train_test_split(train_valid, train_size=0.9, test_size = None)
test_list = pd.DataFrame(sorted(test))
train_list = pd.DataFrame(sorted(train))
valid_list = pd.DataFrame(sorted(valid))
test_list.to_csv(save_dir + 'test.csv', index = False, header = False)
train_list.to_csv(save_dir + 'train.csv', index = False, header = False)
valid_list.to_csv(save_dir + 'valid.csv', index = False, header = False)

test_list.to_csv(save_dir + 'test.csv.gz', index = False, header = False, compression='gzip')
train_list.to_csv(save_dir + 'train.csv.gz', index = False, header = False, compression='gzip')
valid_list.to_csv(save_dir + 'valid.csv.gz', index = False, header = False, compression='gzip')


# In[55]:


import torch
print(torch.__version__)


# In[56]:


import torch_geometric as pyg
print(pyg.__version__)


# In[57]:


print(nx.__version__)

