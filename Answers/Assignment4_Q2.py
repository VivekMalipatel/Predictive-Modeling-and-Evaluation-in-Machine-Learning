# -*- coding: utf-8 -*-
"""
@Name: Assignment_4_Q2.py
@creation Date: March 23, 2022
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import itertools
import matplotlib.pyplot as plt
import numpy
import pandas
import sys
import time

from sklearn.neural_network import MLPRegressor
import Utility

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.7}'.format

catName = ['f_primary_age_tier', 'f_primary_gender', 'f_marital', 'f_residence_location', \
           'f_fire_alarm_type', 'f_mile_fire_station', 'f_aoi_tier']

nPredictor = len(catName)

inputData = pandas.read_excel('Homeowner_Claim_History.xlsx',sheet_name = 'HOCLAIMDATA')

# Create Severity
yName = 'Frequency'

inputData[yName] = numpy.where(inputData['exposure'] > 0.0, inputData['num_claims'] / inputData['exposure'], numpy.NaN)

trainData = inputData[catName + [yName]].dropna().reset_index(drop = True)

# Reorder the categories of the categorical variables in ascending frequency
for pred in catName:
    u = trainData[pred].astype('category').copy()
    u_freq = u.value_counts(ascending = True)
    trainData[pred] = u.cat.reorder_categories(list(u_freq.index)).copy()

# Generate the dummy variables
X0 = pandas.get_dummies(trainData[catName].astype('category'))
X0.insert(0, '_BIAS_', 1.0)

# Identify the aliased parameters
n_param = X0.shape[1]
XtX = X0.transpose().dot(X0)
origDiag = numpy.diag(XtX)
XtXGinv, aliasParam, nonAliasParam = Utility.SWEEPOperator (n_param, XtX, origDiag, sweepCol = range(n_param), tol = 1.0e-7)
X_train = X0.iloc[:, list(nonAliasParam)].drop(columns = ['_BIAS_'])

y_train = trainData[yName].copy()

# Grid Search for the best neural network architecture
result = pandas.DataFrame()
actFunc = ['tanh', 'identity']
nLayer = [1,2,3,4,5,6,7,8,9,10]
nHiddenNeuron = [1,2,3,4,5]

maxIter = 10000
randSeed = 31010

combList = itertools.product(actFunc, nLayer, nHiddenNeuron)

# Grid Search for the best neural network architecture
result = []
mse0 = numpy.var(y_train, ddof = 0)

for comb in combList:
   print(comb)

   time_begin = time.time()
   actFunc = comb[0]
   nLayer = comb[1]
   nHiddenNeuron = comb[2]

   nnObj = MLPRegressor(hidden_layer_sizes = (nHiddenNeuron,)*nLayer, \
                        activation = actFunc, verbose = False, \
                        max_iter = maxIter, random_state = randSeed)
   thisFit = nnObj.fit(X_train, y_train)
   y_pred = nnObj.predict(X_train)

   # Simple Residual
   y_simple_residual = y_train - y_pred

   # Root Mean Squared Error
   mse = numpy.mean(numpy.power(y_simple_residual, 2))
   rmse = numpy.sqrt(mse)

   # Relative Error
   relerr = mse / mse0

   # R-Squared
   corr_matrix = numpy.corrcoef(y_train, y_pred)
   pearson_corr = corr_matrix[0,1]

   time_end = time.time()
   time_elapsed = time_end - time_begin

   result.append([actFunc, nLayer, nHiddenNeuron, thisFit.n_iter_, thisFit.best_loss_, \
                  rmse, relerr, pearson_corr, time_elapsed])

result_df = pandas.DataFrame(result, columns = ['Activation Function', 'nLayer', 'nHiddenNeuron', \
               'N Iteration', 'Loss', 'RMSE', 'RelErr', 'Pearson Corr', 'Time Elapsed'])

print(result_df)

'''
# Locate the optimal architecture
ipos = numpy.argmin(result_df['RMSE'])
row = result_df.iloc[ipos]

actFunc = row['Activation Function']
nLayer = row['nLayer']
nHiddenNeuron = row['nHiddenNeuron']

print('=== Optimal Model ===')
print('Activation Function: ', actFunc)
print('Number of Layers: ', nLayer)
print('Number of Neurons: ', nHiddenNeuron)

# Final model
nnObj = MLPRegressor(hidden_layer_sizes = (nHiddenNeuron,)*nLayer, \
                     activation = actFunc, verbose = False, \
                     max_iter = maxIter, random_state = randSeed)

thisFit = nnObj.fit(X_train, y_train)
y_pred = nnObj.predict(X_train)

# Simple Residual
y_simple_residual = y_train - y_pred

# Pearson Residual
y_pearson_residual = y_simple_residual / numpy.sqrt(y_pred)

# Root Mean Squared Error
mse = numpy.mean(numpy.power(y_simple_residual, 2))
rmse = numpy.sqrt(mse)

# Relative Error
relerr = mse / numpy.var(y_train, ddof = 0)

# R-Squared
corr_matrix = numpy.corrcoef(y_train, y_pred)
pearson_corr = corr_matrix[0,1]

# Plot predicted severity versus observed severity
fig, ax0 = plt.subplots(nrows = 1, ncols = 1, dpi = 200)
ax0.scatter(y_train, y_pred, c = 'royalblue', marker = 'o', s = 10)
ax0.set_xlabel('Observed Frequency')
ax0.set_ylabel('Predicted Frequency')
ax0.set_xticks(numpy.arange(0.0, 220.0, 20.0))
ax0.set_yticks(numpy.arange(0.0, 14.0, 2.0))
ax0.xaxis.grid(True)
ax0.yaxis.grid(True)
plt.show()

fig, (ax0, ax1) = plt.subplots(nrows = 1, ncols = 2, dpi = 200, sharex = True, figsize = (12,6))

# Plot simple residuals versus observed severity
ax0.scatter(y_train, y_simple_residual, c = 'royalblue', marker = 'o')
ax0.set_xlabel('Observed Frequency')
ax0.set_ylabel('Simple Residual')
ax0.set_xticks(numpy.arange(0.0, 220.0, 20.0))
ax0.xaxis.grid(True)
ax0.yaxis.grid(True)

# Plot Pearson residuals versus observed severity
ax1.scatter(y_train, y_pearson_residual, c = 'royalblue', marker = 'o')
ax1.set_xlabel('Observed Frequency')
ax1.set_ylabel('Pearson Residual')
ax1.set_xticks(numpy.arange(0.0, 220.0, 20.0))
ax1.xaxis.grid(True)
ax1.yaxis.grid(True)

plt.show()
'''
