# README -- Gamora: Graph Learning based Symbolic Reasoning for Large-Scale Boolean Networks (DAC'23)

#### 1) Data generator 


```python
dataset_prep/dataset_generator.py
class ABCGenDataset
```

- gentype =0 CSA-array Multiplier generation and labeling
- gentype =1 CPA Adder generation and labeling
- gentype =2 Read a design and generate dataset 
- gentype =3 Generate Booth-encoded multiplier (tbd)

Note: ABC is required (../abc) (make sure to create a link symbol of abc binary
in this folder)



#### 2) Train-Test Demo - training on 8-bit CSA and predicting on 32-bit CSA

```python
python gnn_sampler.py --datagen 0 --bits 8 --datagen_test 0 --bits_test 32 --multilabel True --num_class 6
```
Multilabel sampling and data generation; trained with 6 classes for initial testing


ABC implementation for data generation.
```python
class Gia_EdgelistMultiLabel()
```
 
