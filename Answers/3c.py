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

min_severity_row = trainData.loc[trainData['Severity'].idxmin()]
max_severity_row = trainData.loc[trainData['Severity'].idxmax()]

print()
print("Minimum Severity Combination : ")
print(min_severity_row )
print()
print("Maximum Severity Combination : ")
print(max_severity_row )