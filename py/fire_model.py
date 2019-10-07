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
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


class comparisons:
    def __init__(self, error, acc):
        """
        Note:
            Creates a 3x3 grid in the shape of a numpy array to later generate a density plot.

        Args:
            error: dict containing numpy array
                 Contains all the sources that have been misclassified.
            acc: dict carrying numpy array
                 Holds all the sources that have been correctly classified

        Returns:
            grid: numpy array (3x3)
                 Holds the density of each combination.
                 Rows and columns represent 1 2 3 for classification.
                 For the x=y line (diagonal from bottom left to top right)
                      it indicates correct classification.
                 For all other ones it indicates incorrect classification

        """
        error_labels = sorted(error)
        actual_labels = sorted(acc)
        grid = np.zeros((3,3))
        for i in error_labels:
            grid[int(i[0])-1][int(i[1])-1] = len(error[i])
        for i in actual_labels:
            grid[int(i) - 1][int(i) - 1] = len(acc[i])
        grid = np.transpose(grid)
        grid = np.flipud(grid)
        self.grid = grid

    def graph(self,data1, data2): #, data3, data4):
        """
        Note:
            Plots the 3x3 arrays as images to show a 2D plot with density color bar

        Args:
            data(1, 2): numpy array (3x3)
                 See above in grid      

        Returns:
            2 subplot graphs:
                 Color coded graphs that indicate the density of each combination (1-1, 1-2, etc.).
                 Values are reduced to fractions of the whole.
        """
        f, (ax1,ax2) = plt.subplots(1,
                                    2, squeeze = True, figsize=(15,10))

        #This normalizes the data into creating "probability" plots
        data = [data1, data2]
        data1 = data1/np.sum(data1)
        data2 = data2/np.sum(data2)


        #Graphs the first subplot in the top left 
        axis1 = ax1.imshow(
            data1, norm = colors.LogNorm(vmin = 0.01, vmax = 1),
            interpolation = 'nearest', cmap = 'gray')
        ax1.set_title('Carbon v Non-Carbon')
        ax1.set_xlabel('Prediction')
        ax1.set_ylabel('Actual')
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(axis1, cax=cax1)

        #Graphs the first subplot in the top right 
        axis2 = ax2.imshow(
            data2, norm = colors.LogNorm(vmin = 0.01, vmax = 1),
            interpolation = 'nearest', cmap = 'gray')
        ax2.set_title('Oxygen v Super Giant')
        ax2.set_xlabel('Prediction')
        ax2.set_ylabel('Actual')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(axis2, cax=cax2)


        #Adds tick marks with labels to each subplot.
        axes = [ax1, ax2]
        axis = [axis1, axis2]
        for i in axes:
            plt.setp(i, xticks= [0,1,2], xticklabels=['1','2','3'],
                     yticks=np.arange(2,-1, -1), yticklabels = ['1','2','3'])

        plt.show()
