# README -- Gamora: Graph Learning based Symbolic Reasoning for Large-Scale Boolean Networks (DAC'23)

#### 1) Data generator 

```python
dataset_prep/dataset_generator.py
class ABCGenDataset
```

Functionalities (gentype)

```python
    def __init__(self, bits = 8, gentype = 0, root = '', designfile = ''):
```

- gentype =0 CSA-array Multiplier generation and labeling
- gentype =1 CPA Adder generation and labeling
- gentype =2 Read a design and generate dataset 
- gentype =3 Generate Booth-encoded multiplier (tbd)


#### 2) Usage in gnn_sampler.py (dataset gen + training/testing)

Generate multiplier with bitwidth:
```python
python gnn_sampler.py --datagen 0 --bits 13
``` 

Generate adder with bitwidth:
```python
python gnn_sampler.py --datagen 1 --bits 15
``` 

Load design:
```python
python gnn_sampler.py --datagen 2 -f your_design.blif
``` 


#### 3) Demo

```python
python gnn_sampler.py --datagen 0 --bits 8 --datagen_test 0 --bits_test 32 --multilabel True --num_class 6
```
Multilabel sampling and data generation; trained with 6 classes for initial testing

```python
class Gia_EdgelistMultiLabel()
```
ABC updates for multilabel generation.
 
