# Gamora: Graph Learning based Symbolic Reasoning for Large-Scale Boolean Networks (DAC'23) 

## Installation

### Prereq: 
1) Python packages: torch and torch_geometric 
	

2) Package: readline (follow ABC installation requirement)
	
	sudo apt-get install libreadline6 libreadline6-dev (Ubuntu)

	sudo yum install readline-devel (CentOS, RedHat)

### Installation
compile ABC customized for graph learning
	
	cd abc;make clean;make -j4



## Implementation

#### 1) Data generator 


```python
dataset_prep/dataset_generator.py
class ABCGenDataset
```

- gentype =0 CSA-array Multiplier generation and labeling
- gentype =1 CPA Adder generation and labeling
- gentype =2 Read a design and generate dataset
- gentype =3 Generate Booth-encoded multiplier (tbd)

**Note: ABC is required (../abc) (make sure to create a link symbol of abc binary
in this folder)**

```bash
ln -s ../abc/abc .
```


***ABC implementation for data generation.***

```c
// abc/src/proof/acec/acecXor.c
class Gia_EdgelistMultiLabel()
```



#### 2) Train-Test Demo - training on 8-bit CSA and predicting on 32-bit CSA

```python
python gnn_sampler.py --datagen 0 --bits 8 --datagen_test 0 --bits_test 32 --multilabel True --num_class 6
```

*Training INPUT*: 8-bit CSA-Mult

*Testing INPUT*: 32-bit CSA-Mult


```bash
# training
Highest Train: 96.60
Highest Valid: 95.79
  Final Train: 96.60
   Final Test: 96.44

# testing
mult32
Highest Train: 0.00 ± nan
Highest Valid: 0.00 ± nan
  Final Train: 0.00 ± nan
   Final Test: 98.77 ± nan
```



#### New commands for ABC

```bash
	abc 01> edgelist -h
	usage: edgelist : Generate pre-dataset for graph learning (MPNN,GraphSAGE, dense graph matrix)
	-F : Edgelist file name (*.el)
	-c : Class map for corresponding edgelist (Only for GraphSAGE; must has -F -c -f all enabled)
 	-f : Features of nodes (Only for GraphSAGE; must has -F -c -f all enabled)
 	-L : Switch to logic netlist without labels (such as AIG and LUT-netlist)
 	Example 1 (GraphSAGE dataset)
 		 read your.aig; edgelist -F test.el -c test-class-map.json -f test-feats.csv 
 	Example 2 (Generate dataset for LUT-mapping netlist; unsupervised)
 		  read your.blif; strash; if -K 6; edgelist -L -F lut-test.el 
 	Example 3 (Generate dataset for abstraction; supervised for FA/HA extraction  - GraphSAGE)
 		  read your.blif; strash; &get; &edgelist -F test.el -c test-class_map.json -f test-feats.csv
```
