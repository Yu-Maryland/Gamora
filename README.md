# Gamora: Graph Learning based Symbolic Reasoning for Large-Scale Boolean Networks (DAC'23) 

## Installation

### Prereq: 
1) Python packages: 
	

2) Package: readline (follow ABC installation requirement)
	
	sudo apt-get install libreadline6 libreadline6-dev (Ubuntu)
	sudo yum install readline-devel (CentOS, RedHat)

### Installation
compile ABC customized for graph learning
	
	cd abc;make clean;make -j4

## New commands for ABC
### Task 1: Word-level reasoning
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

