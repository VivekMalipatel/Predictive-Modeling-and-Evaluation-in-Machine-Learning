import numpy
import pandas
import time
import itertools
from sklearn.neural_network import MLPRegressor
import Utility
import warnings
warnings.filterwarnings("ignore")

catName = ['f_primary_age_tier', 'f_primary_gender', 'f_marital', 'f_residence_location', 'f_fire_alarm_type', 'f_mile_fire_station', 'f_aoi_tier']

nPredictor = len(catName)

inputData = pandas.read_excel('Homeowner_Claim_History.xlsx',sheet_name = 'HOCLAIMDATA')

# Create Severity
yName = 'Severity'

inputData[yName] = numpy.where(inputData['num_claims'] > 0, inputData['amt_claims'] / inputData['num_claims'], numpy.NaN)
inputData[yName] = numpy.log(inputData[yName])

trainData = inputData[['policy'] +catName + [yName]].dropna().reset_index(drop = True)
trainData.set_index('policy', inplace=True)

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
X_reduce = X0.iloc[:, list(nonAliasParam)].drop(columns = ['_BIAS_'])
y_reduce = trainData[yName].copy()

X_train = X_reduce[X_reduce.index.str[0].isin(['A', 'G', 'Z'])]
y_train = y_reduce[y_reduce.index.str[0].isin(['A', 'G', 'Z'])]

X_test = X_reduce[~X_reduce.index.str[0].isin(['A', 'G', 'Z'])]
y_test = y_reduce[~y_reduce.index.str[0].isin(['A', 'G', 'Z'])]

# Grid Search for the best neural network architecture
result = pandas.DataFrame()
actFunc = ['tanh', 'identity', 'relu']
nLayer = [1,2,3,4,5,6,7,8,9,10]
nHiddenNeuron = [1,2,3,4,5]

maxIter = 10000
randSeed = 2023484

combList = itertools.product(actFunc, nLayer, nHiddenNeuron)

# Grid Search for the best neural network architecture
result = []
mse0 = numpy.var(y_test, ddof = 0)

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
   y_pred = nnObj.predict(X_test)

   # Simple Residual
   y_simple_residual = y_test - y_pred

   # Root Mean Squared Error
   mse = numpy.mean(numpy.power(y_simple_residual, 2))
   rmse = numpy.sqrt(mse)

   # Relative Error
   relerr = mse / mse0

   # R-Squared
   corr_matrix = numpy.corrcoef(y_test, y_pred)
   pearson_corr = corr_matrix[0,1]

   time_end = time.time()
   time_elapsed = time_end - time_begin

   result.append([actFunc, nLayer, nHiddenNeuron, thisFit.n_iter_, thisFit.best_loss_, \
                  rmse, relerr, pearson_corr, time_elapsed])

result_df = pandas.DataFrame(result, columns = ['Activation Function', 'nLayer', 'nHiddenNeuron', \
               'N Iteration', 'Loss', 'RMSE', 'RelErr', 'Pearson Corr', 'Time Elapsed'])

print(result_df)
result_df.to_csv('3aOutput.csv')