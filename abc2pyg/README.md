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

* Dataset generation
    ```python
    python ABC_dataset_generation.py --bits 8
    # generate an 8-bit CSA multiplier
    ```
    ```python
    python ABC_dataset_generation.py --bits 32
    # generate a 32-bit CSA multiplier
    ```

* Training and inference
    ```python
    python gnn_multitask.py --bits 8 --bits_test 32
    # training with mult8, and testing with mult32
    ```
    * An updated version with further speedup (suggested environment: [PyG 23.07 container from NVIDIA](https://developer.nvidia.com/pyg-container-early-access))
      ```python
    	python gnn_multitask_v2.py --bits 8 --bits_test 32
    	# training with mult8, and testing with mult32
      ```

* Inference with pre-trained model
    ```python
    python gnn_multitask_inference.py --model_path SAGE_mult8 --bits_test 32 --design_copies 1
    # load the pre-trained model "SAGE_mult8", and test with mult32
    ```

*Training INPUT*: 8-bit CSA-Mult

*Testing INPUT*: 32-bit CSA-Mult


```bash
# training
Highest Train: 99.45
Highest Valid: 100.00
  Final Train: 98.90
   Final Test: 99.12

# testing
mult32
Highest Train: 0.00 ± nan
Highest Valid: 0.00 ± nan
  Final Train: 0.00 ± nan
   Final Test: 99.95 ± nan
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
