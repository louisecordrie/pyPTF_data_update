import os
import sys
import h5py
import numpy as np
import pathlib
import hickle as hkl
import json
import pprint

### Creation dictionnary for Focal Mechanism ###

fm = {}

fm['2003_0521_boumardes']= np.array([ [57,44,71],
                                      [262,49,107],
                                      [ 54, 47, 88],
                                      [237, 43, 92],
                                      [ 65, 27, 86],
                                      [ 250, 63, 92],
                                      [ 62, 25, 82],
                                      [ 251, 65, 94],
                                      [ 75, 30, 98],
                                      [ 246, 61, 85] ])

fm['2015_0416_crete']= np.array([ [56,43,21],
                                  [310,76,131],
                                  [51,51,26],
                                  [304,70,138],
                                  [293,54,142],
                                  [47,60,43],
                                  [300,64,142],
                                  [48,56,31],
                                  [71,51,68],
                                  [284,44,115],
                                  [57,85,349],
                                  [148,79,185]])

fm['2015_1117_lefkada'] = np.array([ [22,64,179], 
                                     [113,89,26],
                                     [23,71,179],
                                     [113,89,19],
                                     [112,86,6],
                                     [21,83,176],
                                     [21,85,155],
                                     [113,65,6],
                                     [292,84,343],
                                     [24,73,187]])

fm['2016_0125_gibraltar'] = np.array([ [120,73,166], 
                                       [214,76,17],
                                       [211,75,10],
                                       [119,80,165],
                                       [121,89,181],
                                       [31,89,0],
                                       [302,62,187],
                                       [209,84,332],
                                       [303,89,161],
                                       [33,71,1]])

fm['2016_1030_norcia'] = np.array([ [154,37,264],
                                    [342,53,275],
                                    [155,37,262],
                                    [345,53,276],
                                    [151,44,266],
                                    [338,45,275],
                                    [162,27,276],
                                    [335,63,267],
                                    [168,32,282],
                                    [334,59,263]])

fm['2017_0612_lesbo'] = np.array([ [286,43,267],
                                   [110,47,273],
                                   [84,33,229],
                                   [311,66,293],
                                   [111,41,276],
                                   [284,48,266],
                                   [279,33,258],
                                   [114,57,278],
                                   [118,41,286],
                                   [277,51,256]])

fm['2017_0720_kos-bodrum']=np.array([ [ 278,36,278],
                                      [ 88,55,264],
                                      [ 296,49,305], 
                                      [ 68,52,236],
                                      [ 285, 39, 287],
                                      [ 84, 53, 257],
                                      [ 275, 36, 275],
                                      [ 89, 54, 266],
                                      [ 270, 56, 266],
                                      [ 97, 34, 276],
                                      [ 286, 53, 288],
                                      [ 78, 41, 248] ])

fm['2018_1025_zante'] = np.array([[11,28,165],
                                 [114,83,63],
                                 [17,27,168],
                                 [117,85,63],
                                 [109,82,52],
                                 [9,39,167],
                                 [108,85,41],
                                 [14,49,174],
                                 [107,85,68],
                                 [5,23,167],
                                 [117,85,63],
                                 [17,27,168]])


fm['2019_0320_turkey'] = np.array([ [321,42,273], 
                                    [137,48,267], 
                                    [330,34,281],
                                    [136,57,262],
                                    [143,45,280],
                                    [309,45,260],
                                    [326,50,273],
                                    [140,40,266]])

fm['2019_0921_albania'] = np.array([ [336,31,112],
                                     [130,62,77],
                                     [328,37,95], 
                                     [142,53,86],
                                     [132,52,85],
                                     [319,37,95],
                                     [323,32,93],
                                     [139,58,88]])


fm['2019_1126_albania'] = np.array([ [351,25,114],
                                     [145,68,79],
                                     [351,22,115],
                                     [145,71,80],
                                     [150,72,88],
                                     [335,17,94],
                                     [338,27,92],
                                     [156,63,89],
                                     [354,19,113],	
                                     [150,72,82]])

fm['2020_0502_crete'] = np.array([[257,24,71],
                                 [97,68,98],
                                 [273,26,94],
                                 [89,64,88],
                                 [264,22,76],
                                 [98,67,95],
                                 [229,31,46],
                                 [98,68,113],
                                 [254,33,67],
                                 [101,60,105]])


fm['2020_1030_samos']=np.array([[270,37,265],
                               [96,53,274],
                               [275,29,272],
                               [93,61,269],
                               [272,48,267],
                               [97,41,275],
                               [272,55,267],
                               [97,34,275],
                               [275,45,264],
                               [103,45,275],
                               [289,40,291],
                               [82,53,253],
                               [260,36,244],
                               [111,58,288],
                               [270,46,269],
                               [95,43,273],
                               [294,54,295],
                               [76,43,240]])


fm['2023_0206_turkey'] = np.array([[54,70,11],
                                 [320,80,160],
                                 [233,74,18],
                                 [140,77,168],
                                 [318,89,181],
                                 [228,89,359]])


h5file = 'focal_mech' + '.hdf5'
hkl.dump(fm, h5file, mode='w')

### Creation dictionnary for Tsunami data ###

tsu = {}

tsu['2003_0521_boumardes']=np.array([[1.44, 38.91, 0.593],
                         [2.16, 41.35, 0.430],
                         [-4.41, 36.71, 0.242],
                         [-0.33, 39.46, 0.666],
                         [2.62, 39.55, 1.157],
                         [1.30, 38.97, 1.960],
                         [3.00, 36.75, 0.217],
                         [8.76, 41.93, 0.124],
                         [5.36, 43.29, 0.049],
                         [7.28, 43.69, 0.124],
                         [5.90, 43.12, 0.194],
                         [9.10, 39.22, 0.242],
                         [8.03, 43.88, 0.058],
                         [8.33, 39.15, 0.280],
                         [13.33, 38.13, 0.066],
                         [10.30, 43.55, 0.247],
                         [8.38, 40.83, 0.00],
                         [13.52, 37.28, 0.275],
                         [8.92, 44.42, 0.122]])


tsu['2015_0416_crete']=np.array([[ 27.424,37.032,0.0082415],
                                 [ 24.12,34.846,0],
                                 [ 23.637,35.514,0]])


tsu['2015_1117_lefkada']=np.array([[17.130,39.084,0.22522],
                                   [18.497,40.147,0],
                                   [17.224,40.476,0],
                                   [22.999,36.138,0],
                                   [21.323,37.644,0],
                                   [21.963,36.797,0]])

tsu['2016_0125_gibraltar']=np.array([[-5.45,36.13,0],
                                     [-2.468,36.84,0],
                                     [-1.8996,36.974,0],
                                     [-0.98,37.606,0],
                                     [-5.367,36.133,0],
                                     [-4.421,36.72,0],
                                     [-2.94,35.3,0],
                                     [-3.5236,36.72,0],
                                     [-5.6036,36.006,0]])

tsu['2016_1030_norcia']=np.array([[13.507,43.625,0],
                                  [11.79,42.094,0],
                                  [13.59,41.21,0],
                                  [10.299,43.546,0],
                                  [10.238,42.743,0],
                                  [15.275,40.03,0],
                                  [14.415,42.356,0],
                                  [16.177,41.888,0],
                                  [12.283,44.492,0],
                                  [13.89,42.955,0],
                                  [13.758,45.649,0],
                                  [12.427,45.418,0],
                                  [8.7817,41.934,0],
                                  [9.3501,42.967,0],
                                  [9.4039,41.826,0]])

tsu['2017_0612_lesbo']=np.array([[26.92,35.42,0],
                                 [22.42,38.43,0],
                                 [22.08,38.26,0],
                                 [27.424,37.032,0],
                                 [25.894,40.232,0],
                                 [23.621,37.935,0],
                                 [24.941,37.438,0]])


tsu['2017_0720_kos-bodrum']=np.array([[26.92,35.42,0.14164],
                                      [25.743,35.009,0.0],
                                      [27.424,37.032,0.25026],
                                      [25.894,40.232,0.0],
                                      [23.621,37.935,0.0],
                                      [24.941,37.438,0.033026],
                                      [27.275,36.828,0.6],
                                      [27.300, 36.973, 0.700],
                                      [27.312, 36.980, 0.500],
                                      [27.331, 37.014, 0.250],
                                      [27.384, 37.026, 0.250],
                                      [27.405, 37.031, 0.950],
                                      [27.425, 37.036, 0.550],
                                      [27.256, 37.006, 0.300],
                                      [27.279, 36.966, 0.450],
                                      [27.286, 36.961, 0.250],
                                      [27.300, 36.970, 0.800],
                                      [27.301, 36.974, 0.700],
                                      [27.304, 36.976, 0.450],
                                      [27.373, 37.027, 0.450],
                                      [27.384, 37.025, 0.250],
                                      [27.404, 37.032, 0.575],
                                      [27.417, 37.028, 0.500],
                                      [27.461, 36.966, 0.875],
                                      [28.324, 37.051, 0.300],
                                      [27.207, 36.890, 0.025],
                                      [27.238, 36.897, 0.050],
                                      [27.258, 36.904, 0.250],
                                      [27.273, 36.912, 0.750],
                                      [27.275, 36.914, 0.700],
                                      [27.276, 36.914, 0.350],
                                      [27.280, 36.916, 0.300],
                                      [27.281, 36.915, 0.300],
                                      [27.283, 36.911, 0.300],
                                      [27.285, 36.906, 0.150],
                                      [27.286, 36.901, 0.600],
                                      [27.288, 36.896, 0.750],
                                      [27.294, 36.893, 0.350],
                                      [27.301, 36.893, 0.400],
                                      [27.305, 36.888, 0.350],
                                      [27.319, 36.885, 0.400],
                                      [27.326, 36.885, 0.400],
                                      [27.332, 36.887, 0.300],
                                      [27.339, 36.890, 0.250],
                                      [27.346, 36.882, 0.400],
                                      [27.352, 36.871, 0.350]])

tsu['2018_1025_zante']=np.array([[17.13,39.084,0.21386],
                                [22.42,38.43,0],
                                [22.08,38.26,0],
                                [21.655,37.258,0.19639],
                                [18.497,40.147,0.13964],
                                [22.11,37.022,0],
                                [21.323,37.644,0.20621],
                                [17.029,38.908,0.12912],
                                [23.621,37.935,0],
                                [24.941,37.438,0]])

tsu['2019_0320_turkey']=np.array([[32.94,36.096,0]])

tsu['2019_0921_albania']=np.array([[16.866,41.14,0],
                                   [17.13,39.084,0],
                                   [15.502,42.119,0],
                                   [22.91,40.63,0],
                                   [18.497,40.147,0],
                                   [15.275,40.03,0],
                                   [17.224,40.476,0],
                                   [16.177,41.888,0]])

tsu['2019_1126_albania']=np.array([[16.866,41.14,0],
                                   [17.13,39.084,0],
                                   [15.502,42.119,0],
                                   [22.91,40.63,0],
                                   [18.497,40.147,0],
                                   [15.275,40.03,0],
                                   [17.224,40.476,0],
                                   [16.177,41.888,0]])

tsu['2020_0502_crete']=np.array([[26.92,35.42,0.11629],
                                 [25.743,35.009,0.33361],
                                 [27.424,37.032,0],
                                 [29.092,36.621,0],
                                 [24.941,37.438,0]])

tsu['2020_1030_samos']=np.array([[25.15,35.35,0.16125],
                                 [27.424,37.032,0.061528],
                                 [27.333,36.906,0.21028],
                                 [27.333,36.906,0.18315],
                                 [26.317,38.977,0.083977],
                                 [24.941,37.438,0.1583],
                                 [26.975,37.756,2],
                                 [26.687,37.796,1.7],
                                 [25.569,37.014,0.5],
                                 [26.945,37.691,0.5],
                                 [26.296,37.614,1],
                                 [26.043,38.201,1.1],
                                 [26.546,37.323,0.5],
                                 [24.935,37.843,0.6],
                                 [26.489,38.196,1.9],
                                 [26.377,38.271,0.3],
                                 [26.694,38.21,0.7],
                                 [26.721,38.215,0.15],
                                 [26.787,38.195,1.22],
                                 [26.786,38.201,1.5],
                                 [26.815,38.164,1.9],
                                 [26.825,38.156,0.8],
                                 [26.821,38.161,0.2],
                                 [26.812,38.166,0.8],
                                 [26.999,38.065,0.5],
                                 [26.833,38.136,1.5],
                                 [26.977,37.753,2],
                                 [26.976,37.756,1.5]])

tsu['2023_0206_turkey'] = np.array([[36.176769256592,36.594230651855,0.15],
                                    [34.32793045044,36.61111831665,0.17],
                                    [34.8850103,32.4714063,0.0],
                                    [33.836227416992,36.281463623047,0.02],
                                    [30.612609863281,36.835536956787,0.0],
                                    [28.384845733643,36.837989807129,0.0],
                                    [27.8575,35.9273,0.0],
                                    [26.92184,35.4186,0.0],
                                    [25.73852921,35.00374985,0.0],
                                    [28.384845733643,36.837989807129,0.0],
                                    [39.744483947754,41.001949310303,0.0]])


h5file = 'tsunami_data' + '.hdf5'
hkl.dump(tsu, h5file, mode='w')