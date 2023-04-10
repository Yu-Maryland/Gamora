### script for writing meta information of datasets into master.csv
### for node property prediction datasets.
import pandas as pd

dataset_dict = {}
dataset_list = []

### add meta-information about paper venue prediction task
name = 'mult8'
dataset_dict[name] = {'num tasks': 1, 'num classes': 5, 'eval metric': 'acc', 'task type': 'multiclass classification'}
dataset_dict[name]['download_name'] = 'mult8'
dataset_dict[name]['version'] = 1
dataset_dict[name]['url'] = None
dataset_dict[name]['add_inverse_edge'] = False 
dataset_dict[name]['has_node_attr'] = True
dataset_dict[name]['has_edge_attr'] = False
dataset_dict[name]['split'] = 'Random'
dataset_dict[name]['additional node files'] = 'None'
dataset_dict[name]['additional edge files'] = 'None'
dataset_dict[name]['is hetero'] = False
dataset_dict[name]['binary'] = False


df = pd.DataFrame(dataset_dict)
# saving the dataframe 
df.to_csv('master.csv')