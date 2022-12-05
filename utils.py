#!/usr/bin/env python3

import numpy

def extract_true_labels(arguments, values):
    true_labels = []
    for arg_index in arguments.index:
        labels = []
        for v in values:
            labels.append(arguments[v][arg_index])
        true_labels.append(labels)
        
    return numpy.array(true_labels)
        