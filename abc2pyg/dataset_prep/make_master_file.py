### script for writing meta information of datasets into master.csv
### for node property prediction datasets.
import pandas as pd
import os.path as osp
import os
import argparse
root_folder = osp.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(root_folder)

def make_master(design_name, num_class=5, new = 0): 
    # new = 0, add new dataset meta info with previous master file
    # new = 1, write a new master file
    master_dir = osp.join(osp.join(root_folder, 'dataset_prep'), 'master.csv')
    if not new and osp.exists(master_dir):
        dataset_dict = pd.read_csv(master_dir, index_col=0).to_dict()
        # print(dataset_dict)
            
        name = design_name
        dataset_dict[name] = {'num tasks': 1, 'num classes': num_class, 'eval metric': 'acc', 'task type': 'multiclass classification'}
        dataset_dict[name]['download_name'] = design_name
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
            
    else:  
        dataset_dict = {}
        dataset_list = []
        ### add meta-information about paper venue prediction task
        name = design_name
        dataset_dict[name] = {'num tasks': 1, 'num classes': num_class, 'eval metric': 'acc', 'task type': 'multiclass classification'}
        dataset_dict[name]['download_name'] = design_name
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
    df.to_csv(root_folder+'/dataset_prep/' + 'master.csv')
    return

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--design_name', type=str, default='')
    parser.add_argument('--new', type=int, default=0)
    parser.add_argument('--num_class', type=int, default=5)
    args = parser.parse_args()
    
    make_master(args.design_name, args.num_class, new = args.new)

if __name__ == "__main__":
    main()
