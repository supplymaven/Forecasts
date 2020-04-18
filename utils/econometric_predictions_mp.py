import os
import django
import sys
import MySQLdb
import math
from datetime import datetime, timedelta
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tools.eval_measures import rmse
from django.conf import settings
from django.db.models import Max
from django import db
from django.contrib.admin.utils import flatten
from genetic_algorithm import *

#db=MySQLdb.connect(host="74.208.137.51", port=3306, user="matt_cremeens", passwd="d&U6cg30", db="timeseries")
#db=MySQLdb.connect(host="localhost", port=3306, user="root", passwd="a6!modern", db="timeseries")
#cursor=db.cursor()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Forecasts.settings')
django.setup()

from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from pmdarima.arima import StepwiseContext
import pandas as pd
import numpy as np
from dateutil.relativedelta import *
from MachineLearning.models import timeseries, series_names, arima_predictions
from multiprocessing import Pool, Process

def make_predictions(this_series):
    # list of names of all series except the target
    column_names=timeseries.objects.values_list('series_title', flat=True).distinct().exclude(series_title=this_series)
    # an empty dataframe with columns that are the x-values in our regression
    df=pd.DataFrame()
    df_dict={}
    for x in column_names:
        #print(list(timeseries.objects.filter(series_title=x).values_list('inx', flat=True)))
        data_list=list(timeseries.objects.filter(series_title=x, observation_date__range=['2017-01-01','2019-12-01']).values_list('inx', flat=True))
        if len(data_list)==36: # number of data points given the date range provided
            df_dict.update({x:data_list})    
    df=pd.DataFrame.from_dict(df_dict)       
    #df=pd.DataFrame(list(timeseries.objects.filter(observation_date__lt='2020-01-01', observation_date__gt='2018-12-31').exclude(series_title=request.POST['timeseries']).values()))
    target=pd.DataFrame({this_series:timeseries.objects.filter(series_title=this_series, observation_date__range=['2017-01-01','2019-12-01']).values_list('inx', flat=True)})
    # use a genetic algorithm to select the independent (explanatory) variables
    num_generations=50
    num_variables=len(df.columns)
    #print(num_variables)
    size_of_chromosome_population=100 # 2^5
    crossover_probability=0.7
    mutation_probability=0.001
    #                                  num_possble_vars, num_possible_combinations_of_vars, max number of independent variables allowed
    chromosome_population=generate_initial_population(num_variables,size_of_chromosome_population, 20)
    #print(chromosome_population)
    min_rmse_test=math.inf
    most_fit_chromosome=''
    for i in range(num_generations):  
        # get fitness of each chromosome
        chromosome_fitness_pairs=[]
        accumulated_fitness=0
        for chromosome in chromosome_population:
            # get all indices in bit string where bit string is a '1'
            # we don't want the first three colums, however: id and frequency and the dependent variable
            indices=[i for i, x in enumerate(chromosome) if x == '1']
            #print(indices)
            # these are the dataframe columns corresponding to those bits in the bit string that are equal to 1
            X=df.iloc[:,indices]        
            X=sm.add_constant(X)
            y=target[this_series]
            X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            #X_train=sm.add_constant(X_train)
            model=sm.OLS(y_train.astype(float), X_train.astype(float)).fit()
            predictions_train=model.predict(X_train.astype(float))
            predictions_test=model.predict(X_test.astype(float))
            rmse_train=round(rmse(y_train.astype(float),predictions_train.astype(float)),3)
            # We will use this as our fitness function in the GA
            rmse_test=round(rmse(y_test.astype(float),predictions_test.astype(float)),10) 
            #population_rmse_test.append(rmse_test)
            #print(rmse_test)
            if rmse_test<min_rmse_test:
                min_rmse_test=rmse_test              
                most_fit_chromosome=chromosome
         
            chromosome_fitness_pairs.append((chromosome,1/rmse_test))
            accumulated_fitness+=1/rmse_test

        # get the fitness ratios    
        chromosome_fitness_pairs=[(i,j/accumulated_fitness) for (i,j) in chromosome_fitness_pairs]  
    
        new_chromosome_population=[]
        #population_rmse_test=[]
        
        while len(new_chromosome_population)<=len(chromosome_population):           
            mating_pair=select_chromosome_pair(chromosome_fitness_pairs)
            new_pair=mate_pair(mating_pair, crossover_probability, mutation_probability)
            new_chromosome_population.append(new_pair[0])
            new_chromosome_population.append(new_pair[1])
        
        
        chromosome_population=[new_chromosome for new_chromosome in new_chromosome_population]
        
        # If the test and train rmse are near each other, we assume neither over nor under fitting. 
        # If the best test rmse is close to that of the population, we assume convergence, and can therefore stop the algorithm early.
        #mean_rmse_test=sum(population_rmse_test)/len(population_rmse_test)
        #if abs(min_rmse_test-mean_rmse_test)/((min_rmse_test+mean_rmse_test)/2)<0.04 and abs(min_rmse_test-rmse_train)/((min_rmse_test+rmse_train)/2)<0.04:
        #    break
        
    # get all indices in bit string where bit string is a '1'
    # we don't want the first three colums, however: id and frequency and the dependent variable
    indices=[i for i, x in enumerate(most_fit_chromosome) if x == '1']
    # these are the dataframe columns corresponding to those bits in the bit string that are equal to 1
    X=df.iloc[:,indices]        
    X=sm.add_constant(X)
    y=target[this_series]
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    model=sm.OLS(y_train.astype(float), X_train.astype(float)).fit()
    predictions_train=model.predict(X_train.astype(float))
    predictions_test=model.predict(X_test.astype(float))
    rmse_train=round(rmse(y_train.astype(float),predictions_train.astype(float)),3)
    # We will use this as our fitness function in the GA
    rmse_test=round(rmse(y_test.astype(float),predictions_test.astype(float)),3) 
    now=datetime.now()
    timestamp=datetime.timestamp(now)
    
    #summ=model.summary()
    
    # predictions
    # the first column is a constant column vector of 1s
    column_names=timeseries.objects.filter(series_title__in=X_train.columns.values[1:]).values_list('series_title', flat=True).distinct()
    # an empty dataframe with columns that are the x-values in our regression
    df=pd.DataFrame()
    df_dict={}
    for x in column_names:
        #print(list(timeseries.objects.filter(series_title=x).values_list('inx', flat=True)))
        data_list=list(arima_predictions.objects.filter(series_title=x, future_date__range=['2020-01-01','2020-12-01']).values_list('inx', flat=True))
        df_dict.update({x:data_list})    

    try:
        df=pd.DataFrame.from_dict(df_dict)        
        df=sm.add_constant(df, has_constant='add')
        predictions_future=model.predict(df.astype(float))
    except:
        print(this_series)
        print(column_names)
        return

    print(this_series)
    print(predictions_future.values.tolist())
    return flatten([this_series, [str(p) for p in predictions_future.values.tolist()]])
        
if __name__=='__main__':
    
    #forecasts_already_made=arima_predictions.objects.values_list('series_title', flat=True).distinct()
    selections_list=series_names.objects.values_list('series_title', flat=True).distinct()[:10]
    p=Pool(4)
    preds=p.map(make_predictions, selections_list)
    #preds=p.imap_unordered(make_predictions, selections_list, 500)
    p.close()
    p.join()
    f=open("econometric_predictions.txt", "w")
    for pred in preds:
        for pr in pred:
            f.write(",".join(str(p) for p in pr)+'\n')
    f.close()
    
    #cursor.close()
    #db.close()
    