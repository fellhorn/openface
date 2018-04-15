#!/usr/bin/env python2
#

import json
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

labelsFile = open('labels.json', 'r')
labels = json.load(labelsFile)
print("labels")
weights = json.load(open('weights.json', 'r'))
print("weights")
featuresFile = open('features.json', 'r')
features = json.load(featuresFile)
print("features")

clf = MLPClassifier(
    solver='lbfgs',
    alpha=0.0001,
    hidden_layer_sizes=(5, 2),
    random_state=1,
    warm_start=True,
    early_stopping=True,
    verbose=True)

clf.fit(features, labels)
print("The score is: {}".format(clf.score(features, labels)))
joblib.dump(clf, 'scikit/mlp.pkl')
