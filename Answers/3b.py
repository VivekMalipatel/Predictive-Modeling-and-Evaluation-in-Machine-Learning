import pandas
import warnings
warnings.filterwarnings("ignore")

result_df = pandas.read_csv('3aOutput.csv')

min_rase_row = result_df.loc[result_df['RMSE'].idxmin()]
actFunca, nLayera, nHiddenNeurona, rasea = min_rase_row[['Activation Function', 'nLayer', 'nHiddenNeuron', 'RMSE']]

print(min_rase_row)