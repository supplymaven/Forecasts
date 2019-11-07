from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import Series
from pandas import concat

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(df, lag=1):
	#df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
    
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)    
    
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]    