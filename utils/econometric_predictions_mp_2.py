import MySQLdb
import math
import random
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tools.eval_measures import rmse
from genetic_algorithm import *
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from pmdarima.arima import StepwiseContext
import pandas as pd
import numpy as np
from dateutil.relativedelta import *
from multiprocessing import Pool, Process

def make_predictions(this_series):
    db=MySQLdb.connect(host="localhost", port=3306, user="root", passwd="a6!modern", db="timeseries")
    cursor=db.cursor()

    # the arima predictions for these indexes did not turn out right so we exclude them for the time being
    to_exclude=['Dow Jones Industrial Average','Gold','PPI Commodity data for Machinery and equipment-Parts and attachments for industrial process furnaces','PPI Commodity data for Nonmetallic mineral products-Handmade pressed and blown glassware not seasona','PPI Commodity data for Textile products and apparel-Womens and girls tailored jackets and vests ex','PPI Commodity data for Transportation equipment-Air brake and other brake equipment not seasonally a','PPI industry data for Industrial process furnace and oven mfg-Parts and attachments for industrial p','PPI industry data for Mechanical power transmission equipment mfg-Plain bearings and bushings not se','PPI industry data for Nonferrous metal (except copper and aluminum) rolling drawing and extruding-Al','PPI industry data for Other metal container mfg-Steel shipping barrels & drums exc. beer barrels (mo','PPI industry data for Power boiler and heat exchanger mfg-Fabricated heat exchangers and steam conde','PPI industry data for Railroad rolling stock mfg-Air brake and other brake equipment not seasonally','PPI industry data for Sawmills-Secondary products not seasonally adjusted','S&P 500','Utility (piped) gas per therm in Atlanta-Sandy Springs-Roswell GA average price not seasonally adjus','Utility (piped) gas service in Atlanta-Sandy Springs-Roswell GA all urban consumers not seasonally a','WTI Crude Oil']
    
    # list of names of all series except the target
    #column_names=list(arima_predictions.objects.values('series_title').annotate(cnt=Count('id')).filter(cnt=12).values_list('series_title', flat=True).distinct().exclude(series_title__in=[this_series]+to_exclude))
    sql=('select series_title ' 
         'from machinelearning_arima_predictions '
         'where series_title not in ("' + '","'.join([this_series]+to_exclude) + '") '
         'group by series_title '
         'having count(id)=12;')
    cursor.execute(sql)
    results=cursor.fetchall()
    column_names=[]
    for result in results:
        column_names.append(result[0])
    
    # an empty dataframe with columns that are the x-values in our regression
    df=pd.DataFrame()
    df_dict={}
    for x in column_names:
        #data_list=list(timeseries.objects.filter(series_title=x, observation_date__range=['2017-01-01','2019-12-01']).values_list('inx', flat=True))
        sql=('select inx '
             'from machinelearning_timeseries '
             'where series_title="' + str(x) + '" and observation_date >= "2017-01-01" and observation_date <= "2019-12-01";')
        cursor.execute(sql)
        results=cursor.fetchall()
        data_list=[]
        for result in results:
            data_list.append(result[0])
        
        
        if len(data_list)!=36:
            continue

        df_dict.update({x:data_list})  			
			
    df=pd.DataFrame.from_dict(df_dict)   
   
    #target_list=list(timeseries.objects.filter(series_title=this_series, observation_date__range=['2017-01-01','2019-12-01']).values_list('inx', flat=True))
    sql=('select inx '
         'from machinelearning_timeseries '
         'where series_title="' + this_series + '" and observation_date >= "2017-01-01" and observation_date <= "2019-12-01";')
    cursor.execute(sql)
    results=cursor.fetchall()
    target_list=[]
    for result in results:
        target_list.append(result[0])
        
    if len(target_list)!=36:
        return
        
    target=pd.DataFrame({this_series:target_list})
    # use a genetic algorithm to select the independent (explanatory) variables
    num_generations=50
    num_variables=len(df.columns)
    size_of_chromosome_population=50 # 2^5
    crossover_probability=0.7
    mutation_probability=0.001
    #                                  num_possble_vars, num_possible_combinations_of_vars, max number of independent variables allowed
    chromosome_population=generate_initial_population(num_variables,size_of_chromosome_population, 10)
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
            
            # getting the p-values as we want to penalize the fitness function in favor of p-values <= 0.1
            df_summary = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0][['coef','P>|t|']]
            p_values=[]
            for i,r in df_summary.iterrows():
                if r[1]>0.1:
                    p_values.append(r[1])
            num_large_p_values=len(p_values)
            penalty_slope=0.17
            penalty=1+penalty_slope*num_large_p_values
            
            
            predictions_train=model.predict(X_train.astype(float))
            predictions_test=model.predict(X_test.astype(float))
            rmse_train=round(rmse(y_train.astype(float),predictions_train.astype(float)),3)
            # We will use this as our fitness function in the GA
            rmse_test=round(rmse(y_test.astype(float),predictions_test.astype(float)),8)*penalty 
            #population_rmse_test.append(rmse_test)
            #print(rmse_test)
            if rmse_test<min_rmse_test:
                min_rmse_test=rmse_test              
                most_fit_chromosome=chromosome
         
            try:
                if rmse_test<0.0001:
                    rmse_test=0.0001
                chromosome_fitness_pairs.append((chromosome,1/rmse_test))
                accumulated_fitness+=1/rmse_test
            except:
                chromosome_fitness_pairs.append((chromosome,1000))
                accumulated_fitness+=1000    

        # get the fitness ratios    
        chromosome_fitness_pairs=[(i,j/accumulated_fitness) for (i,j) in chromosome_fitness_pairs]  
        new_chromosome_population=[]
        while len(new_chromosome_population)<=len(chromosome_population):           
            mating_pair=select_chromosome_pair(chromosome_fitness_pairs)
            new_pair=mate_pair(mating_pair, crossover_probability, mutation_probability)
            if len(new_chromosome_population)==len(chromosome_population):
                break
            new_chromosome_population.append(new_pair[0])
            if len(new_chromosome_population)==len(chromosome_population):
                break
            new_chromosome_population.append(new_pair[1])
 
        chromosome_population=[new_chromosome for new_chromosome in new_chromosome_population]
        
    # get all indices in bit string where bit string is a '1'
    # we don't want the first three colums, however: id and frequency and the dependent variable
    indices=[i for i, x in enumerate(most_fit_chromosome) if x == '1']
    # these are the dataframe columns corresponding to those bits in the bit string that are equal to 1
    X=df.iloc[:,indices]        
    X=sm.add_constant(X)
    y=target[this_series]
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    model=sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

    # predictions
    # the first column is a constant column vector of 1s
    #column_names=list(timeseries.objects.filter(series_title__in=X_train.columns.values[1:]).exclude(series_title__in=to_exclude).values_list('series_title', flat=True).distinct())
    sql=('select series_title '
         'from machinelearning_timeseries '
         'where series_title in ("' + '","'.join(X_train.columns.values[1:]) + '") and series_title not in ("' + '","'.join(to_exclude) + '");')
    
    
    cursor.execute(sql)
    results=cursor.fetchall()
    column_names=[]
    for result in results:
        column_names.append(result[0])
    
    # an empty dataframe with columns that are the x-values in our regression
    df=pd.DataFrame()
    df_dict={}
    for x in column_names:
        #data_list=list(arima_predictions.objects.filter(series_title=x, future_date__range=['2020-01-01','2020-12-01']).values_list('inx', flat=True))
        sql=('select inx '
             'from machinelearning_arima_predictions '
             'where series_title="' + str(x) + '" and future_date >= "2020-01-01" and future_date <= "2020-12-01";')
        
        cursor.execute(sql)
        results=cursor.fetchall()
        data_list=[]
        for result in results:
            data_list.append(result[0])
         
        df_dict.update({x:data_list}) 
        
         
    #try:
    future_dates=['2020-01-01','2020-02-01','2020-03-01','2020-04-01','2020-05-01','2020-06-01','2020-07-01','2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01']
    df_summary = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0][['coef','P>|t|']]

    coefs=[]
    p_values=[]
    indep_vars=[]
    for i,r in df_summary.iterrows():
        indep_vars.append(i)
        coefs.append(r[0])
        p_values.append(r[1])

    df=pd.DataFrame.from_dict(df_dict)        
    df=sm.add_constant(df, has_constant='add')
    predictions_future=model.predict(df.astype(float)).values.tolist()
    
    #series=series_names.objects.get(series_title=this_series)
    sql=('select id '
         'from machinelearning_series_names '
         'where series_title="' + this_series +'";')
     
    cursor.execute(sql)
    results=cursor.fetchone()
    
    series=results[0]
    
    #forecast=econometric_forecast(series=series)
    #forecast.save()
    sql='insert into machinelearning_econometric_forecast (series_id) values (' + str(series) + ');'
    cursor.execute(sql)
    db.commit()
    forecast=cursor.lastrowid 

    
    for i in range(len(coefs)):
        #coeffs=econometric_coefficients(forecasted_series=forecast, coefficients=coefs[i], pvalues=p_values[i], independent_variables=indep_vars[i])
        #coeffs.save()
        sql='insert into machinelearning_econometric_coefficients (independent_variables,coefficients,pvalues,forecasted_series_id) values ("' + str(indep_vars[i]) + '",' + str(coefs[i]) + ',' + str(p_values[i]) + ',' + str(forecast) + ');'
        cursor.execute(sql)
        db.commit()
    for j in range(len(predictions_future)):
        #p=econometric_predictions(future_date=future_dates[j], forecasted_series=forecast, inx=predictions_future[j])
        #p.save()
        sql='insert into machinelearning_econometric_predictions (future_date,inx,forecasted_series_id) values ("' + str(future_dates[j]) + '",' + str(predictions_future[j]) + ',' + str(forecast) + ');'
        cursor.execute(sql)
        db.commit()
        
    cursor.close()
    db.close()
    
if __name__=='__main__':
    
    # only those series titles that have 12 arima predictions
    db=MySQLdb.connect(host="localhost", port=3306, user="root", passwd="a6!modern", db="timeseries")
    cursor=db.cursor()
    #selections_list=list(arima_predictions.objects.values('series_title').annotate(cnt=Count('id')).filter(cnt=12).values_list('series_title', flat=True).distinct())
    sql=('select series_title '
         'from machinelearning_arima_predictions '
         'group by series_title '
         'having count(id)=12;')
    cursor.execute(sql)
    results=cursor.fetchall()
    selections_list=[]
    for result in results:
        selections_list.append(result[0])
        
    selections_list=random.sample(selections_list, k=10)    
        
    cursor.close()
    db.close()
    
    pool=Pool(processes=4)
    pool.map(make_predictions,selections_list)
    pool.close()
    pool.join()
