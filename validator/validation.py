#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import itertools
import os
import json
import csv

import numpy as np
import pandas as pd
np.set_printoptions(precision=2)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

import openface
from sklearn.externals import joblib
import random


def createHistogram(data, title):
    fig, ax = plt.subplots()

    n, bins = np.histogram(data, 50)
    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n

    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = patches.PathPatch(barpath)
    ax.add_patch(patch)

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())
    ax.set_title(title)

    plt.show()


def findDecisionBorder(same, different, step=0.001):
    decisionBoundry = 0.0
    falseNegativeNumber = 0
    falsePositiveNumber = len(different)

    while (float(falseNegativeNumber) / len(same) <
           float(falsePositiveNumber) / len(different)):
        decisionBoundry += step
        falseNegativeNumber = len(filter(lambda x: x < decisionBoundry, same))
        falsePositiveNumber = len(
            filter(lambda x: x > decisionBoundry, different))

    print("Decision @ {}".format(decisionBoundry))
    print("falseNegativeNumber: ", falseNegativeNumber)
    print("FalsePositiveNumber: ", falsePositiveNumber)
    print("lengths {} and {}".format(len(same), len(different)))
    print("compare: {} and {}".format(
        float(falseNegativeNumber) / len(same),
        float(falsePositiveNumber) / len(different)))


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument(
    '--dlibFacePredictor',
    type=str,
    help="Path to dlib's face predictor.",
    default=os.path.join(dlibModelDir,
                         "shape_predictor_68_face_landmarks.dat"))
parser.add_argument(
    '--networkModel',
    type=str,
    help="Path to Torch network model.",
    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument(
    '--imgDim', type=int, help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))


def getRep(imgPath):
    if args.verbose:
        print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print(
            "  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(
        args.imgDim,
        rgbImg,
        bb,
        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print(
            "  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)
    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(
            time.time() - start))
        print("Representation:")
        print(rep)
        print("-----\n")
    return rep


representations = {}
for (img) in (args.imgs):
    try:
        rep = getRep(img)
        filename = img.split('/')[-1]
        print("{};{};{}".format(img, filename, np.sum(rep)))

        representations[filename] = {"rep": rep, "path": img}
    except Exception:
        print("Error getting representations")
        pass

# SHOWING
same = []
different = []
features = []
labels = []
weights = []

clf = joblib.load('scikit/mlp.pkl')
boost = True

with open('pairs.csv', 'rb') as csvfile:
    with open('result3.txt', 'wb') as result:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='"')
        for row in reader:
            if (row[0] in representations) and (row[1] in representations):
                first = representations[row[0]]["rep"]
                second = representations[row[1]]["rep"]
                d = first - second
                l2 = np.dot(d, d)
                if boost and l2 < 1.3:
                    classification = 1
                    proba = 1.0
                else:
                    classification = clf.predict([d])[0]
                    proba = clf.predict_proba([d])[0][1]
                print("{};{}".format(classification, proba))
            else:
                proba = random.random()
                classification = 1 if (proba > 0.5) else 0
                print("{};{};RANDOM!".format(classification, proba))

            result.write("{}\n".format(proba))
#
# decisionBoundry = 1.074
# falseNegative = len(filter(lambda x: x > decisionBoundry, same))
# falsePositive = len(filter(lambda x: x < decisionBoundry, different))
# errorRate = float(falseNegative + falsePositive) / (len(same) + len(different))
#
# print("Error rate: {}".format(errorRate))
