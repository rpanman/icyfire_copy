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
from sklearn import preprocessing


class file_read:
    def __init__(self,fitsfile):
        """
        Note: Class to read out the fits files and output necessary data.

        Args:
             fitsfile: file path
                  Path to get to the fits file to extract data from


       Returns:
             data: astropy table
                  See astropy table for more details on structure
        """
        self.data = Table.read(fitsfile)
        self.tags = self.data.colnames
        

        
class data_to_pytorch:
    def __init__(self, data):
        """
        Note: 
              Class take the fitsfile data and organizes and labels data to different classifications.
              labels=names (OAGB=1, CAGB=2, RSG= 3 because Pytorch can only use int and floats).

        Args:
             data: astropy table
                  Input data to begin calculations
             preprocess: True or False
                  Normalizes the data to mean of 0, 
                  which includes negative numbers.

        Returns:
             self.label: numpy array
                  Contains spectra in flux units
             self.spectra: numpy array
                  Contains labels of the star classification
        """
        #Setting table columns to variables
        self.name = data['Name']
        self.wave = data['Wave']
        self.flux = data['Flux']
        self.dflux = data['Dflux']

        #Outputting all the unique names to relabel them as numbers
        self.name_unique = np.unique(self.name)

    def relabelling(self, labels, preprocess = True):

        for i in self.name_unique:
            self.name[self.name == (i)] = labels[i]


        #Setting to float data type
        name3 = np.array(self.name, dtype = float)

        #Checking to make sure values are actually numbers
        #Then setting as numpy arrays.
        spectra = []
        for i in range(0,len(self.flux)):
            if np.all(np.isfinite(self.flux)[i]):
                spectra.append(self.flux[i])

        label = []
        for i in range(0,len(name3)):
            if np.all(np.isfinite(self.flux)[i]):
                label.append(name3[i])

        label,spectra = np.array(label),np.array(spectra);

        #Normalize the data default to true.
        #Default normalization mean is 0, but translated data 5 units higher.
        if preprocess == True:
            for i in range(0,len(spectra)):
                avg = 5
                spectra[i] = preprocessing.scale(spectra[i])
                spectra[i] = spectra[i] + avg
                
        #Setting global variables
        self.label, self.spectra = label, spectra 
        
    def randomization(self, label, spectra, ratio): 
        """
        Note:
            Randomizes and separates the data according to training or testing

        Args:
            label: numpy array
                 Contains all the labels of the objects (1, 2, 3 in this case)
            spectra: numpy array
                 Holds all the flux values at each wavelength (240x365 in this case)
            ratio:
                 Determines the % of data that is used to train and used to test

        Returns:
            training: Dict containing Torch tensor
                 Contains the remaining fraction of the training data.
            testing: Dict containing Torch tensor
                 Contains the remaining fraction of the labels.
            train: Dict containing Numpy array
                 Contains the remaining fraction of the training data.
            test: Dict containing Numpy array
                 Contains the remaining fraction of the labels.
        """

        #Shuffling the data and then dividing testing from training by a given ratio
        training = {}
        testing = {}
        train = {}
        test = {}

        shuffle_index = np.arange(len(label));
        np.random.shuffle(shuffle_index);

        #mask = [61,144,146]
        #for i in mask: 
        #    shuffle_mask = np.where(shuffle_index == i)
        #    shuffle_index = np.delete( shuffle_index, shuffle_mask)
        #shuffle_index = np.concatenate([[61],[144],[146],shuffle_index])

        spectra_train = np.copy(spectra[shuffle_index[0:len(label)*ratio//100]]);
        label_train  = np.copy(label[shuffle_index[0:(len(label)*ratio//100)]]);
        label_test   = np.copy(label[shuffle_index[len(label)*ratio//100+1:len(label)]]);
        spectra_test = np.copy(spectra[shuffle_index[len(label)*ratio//100+1:len(label)]]);

        #packing the data from np ndarray to torch data
        training_x, training_y = packing_function(spectra_train,label_train)
        testing_x, testing_y  = packing_function(spectra_test,label_test)

        training['x'] = training_x
        training['y'] = training_y
        training['index'] = np.array(shuffle_index[0:len(label)*ratio//100])
        testing['x'] = testing_x
        testing['y'] =testing_y
        testing['index'] = np.array(shuffle_index[len(label)*ratio//100+1:len(label)])

        train['x'] = spectra_train
        train['y'] = label_train
        train['index'] = np.array(shuffle_index[0:len(label)*ratio//100])
        test['x'] = spectra_test
        test['y'] = label_test
        test['index'] = np.array(shuffle_index[len(label)*ratio//100+1:len(label)])
        
        return training, testing, train, test

    
# Simple function to change a numpy array to a torch tensor        
def packing_function(t_x,t_y):
    x=torch.from_numpy(t_x)
    y=torch.from_numpy(t_y)
    return x.float(), y.float()
