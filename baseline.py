#!/usr/bin/env python3

import random
import numpy

parameters = ["Argument ID", "Conclusion", "Stance", "Premise"]

# values = [ "Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face", "Security: personal", "Security: societal", "Tradition", "Conformity: rules", "Conformity: interpersonal", "Humility", "Benevolence: caring", "Benevolence: dependability", "Universalism: concern", "Universalism: nature", "Universalism: tolerance", "Universalism: objectivity" ]

random_probabilities = [ 0.17, 0.26, 0.06, 0.04, 0.27, 0.09, 0.11, 0.07, 0.38, 0.31, 0.11, 0.23, 0.04, 0.08, 0.29, 0.15, 0.38, 0.07, 0.14, 0.18 ]

# "instance" is a dict with keys "Argument ID", "Conclusion", "Stance", and "Premise"
# return value is the list of detected values (here: random)
def label_argument(values):
    argumentValues = []
    for v in range(len(values)):
        if random.random() <= random_probabilities[v]:
            argumentValues.append(0)
        else:
            argumentValues.append(1)
    return argumentValues


def random_prediction(values, arguments):
    print("Random Prediction")
    preds = []
    for i in range(len(arguments)):
        preds.append(label_argument(values))
    
    print(f"Added predictions for {len(arguments)} arguments")
    return numpy.array(preds)


def all_ones(values, arguments):
    print("Predicting all 1")
    preds = []
    for i in range(len(arguments)):
        preds.append([1]*len(values))
    
    print(f"Added predictions for {len(arguments)} arguments")
    return numpy.array(preds)
