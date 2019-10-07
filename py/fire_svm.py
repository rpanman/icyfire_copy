#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:30:52 2018

@authors: bordieremma, panrichard
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys

import torch
from astropy.table import Table
from sklearn import svm, model_selection, preprocessing


#SVM classifier 
class svm_network:
    def __init__(
            self,training_x, training_y,
            testing_x, testing_y = None,
            c = 1, kernel = 'rbf',  gamma = 0.001):

        """
        Note: 
             Class that builds the SVM classifier for training the machine.
             Uses sklearn's support vector machine.
             Questions about c, kernel, and gamma see sklearn's guide for more.

        Args:
             training_x: Torch tensor
                  Contains spectra (flux units) with the same shapes to be trained on.
             training_y: Torch tensor
                  Holds corresponding labels. 
             testing_x: Torch tensor
                  Contains spectra (flux units) with the same shapes to be tested on.
             testing_y: Torch tensor 
                  If classifications are known then set equal to data set. 
                  If not then don't include this parameter.
             c: float
                  Indicates the penalty parameter.  
             kernel: string 
                  Determines decision function. Default set to radial based function. 
                  Options: 'rbf', 'linear', 'sigmoid', 'poly'(Recommend rbf or linear)
             gamma: float
                  Related to c parameter.

        Returns:
             self.label: numpy array
                  Contains spectra in flux units
             self.clf: svm.SVC class
                  Trained machine. To predict data simply do self.clf.predict(testing data)
        """
        self.clf = svm.SVC(C=c, gamma = gamma, kernel = kernel, probability = True)
        self.clf.fit(training_x, training_y)
        self.prediction = self.clf.predict(testing_x)
        self.prediction = torch.from_numpy(self.prediction)

        #Determining the MSE loss (useful for non-binary cases)
        loss_function=torch.nn.MSELoss(size_average=False)
        if testing_y is not None:
            self.loss = loss_function(self.prediction, testing_y)
            
    def confidence(self, testing_x, testing_y):
        """
        Note: 
             Used to output probabilities and determine incorrect predictions.

        Args:
             testing_x: Torch tensor
                  Contains spectra (flux units) with the same shapes to be tested on.
             testing_y: Torch tensor 
                  If classifications are known then set equal to data set. 
                  If not then don't include this parameter.

        Returns:
             self.plist: numpy array
                  Contains the probability distribution of each classification (INACCURATE)
             self.mask: numpy array
                  Holds indices of the incorrect predictions
             self.predict_error: numpy array
                  Carries the labels of the incorrect predictions
             self.actual_error: numpy array
                  Has the labels of the actual classifications
        """
        probs = self.clf.predict_proba(testing_x)
        self.plist = np.zeros([1, np.shape(probs)[1]])
        self.mask = np.where(self.prediction != testing_y)
        if isinstance(len(self.mask), int):
            self.plist = np.vstack((self.plist, probs[self.mask]))
            self.predict_error = self.prediction[self.mask]
            self.actual_error = testing_y[self.mask]
        self.plist = np.delete(self.plist, (0), axis=0)

