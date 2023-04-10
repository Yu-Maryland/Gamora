import argparse
import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator,DirectedGraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from tensorflow.keras import layers, optimizers, losses, metrics, Model

from sklearn import preprocessing, feature_extraction, model_selection
#from IPython.display import display, HTML
#%matplotlib inline
import utils

import pandas as pd
from stellargraph import StellarDiGraph
import numpy as np
from collections import Counter


   
def train(model,args_, generator_train, generator_test,labels_train, labels_test):

    train_labels, val_labels = model_selection.train_test_split(
            labels_train,
            train_size=0.8,
            test_size=None,
            stratify=labels_train,
            random_state=43,
        )
    test_labels = labels_test # 2nd design or design under test (DUT)
    #model_selection.train_test_split(
    #labels_test, train_size=0.01, test_size=None, stratify=labels_test, random_state=88,)
        
    print(len(train_labels),len(val_labels),len(test_labels))
    
    target_encoding = preprocessing.LabelBinarizer()
    train_targets = target_encoding.fit_transform(train_labels)
    val_targets = target_encoding.transform(val_labels)
    test_targets = target_encoding.transform(test_labels)

    train_gen = generator_train.flow(train_labels.index, train_targets, shuffle=True)
    val_gen = generator_test.flow(val_labels.index, val_targets)

    x_inp, x_out = model.in_out_tensors()
    prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
    prediction.shape

    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=args_.lr),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )

    history = model.fit(
        train_gen, epochs=args_.epochs, validation_data=val_gen, verbose=1, shuffle=True )
    sg.utils.plot_history(history)

def main(args_):
    
    G_train,labels_train,size = utils.load_aig("%s.el" % args_.d,
            "%s-feats.csv" % args_.d, "%s-class_map.json" % args_.d)
            
    G_test,labels_test,size = utils.load_aig("%s.el" % args_.d,
            "%s-feats.csv" % args_.dut, "%s-class_map.json" % args_.dut)

    batch_size = args_.batch_size
    num_samples = [args_.s1,args_.s2]
    generator_train = GraphSAGENodeGenerator(G_train, batch_size, num_samples)
    generator_test = GraphSAGENodeGenerator(G_test, batch_size, num_samples)
    
    model = GraphSAGE(layer_sizes=[8, 4], generator=generator_train, bias=True, dropout=0.25 )
    train(model, args_, generator_train, generator_test, labels_train, labels_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--d', type=str, default="mult8")
    parser.add_argument('--dut', type=str, default="mult8")
    parser.add_argument('--s1', type=int, default=10)
    parser.add_argument('--s2', type=int, default=5)
    args_ = parser.parse_args()
    main(args_)
