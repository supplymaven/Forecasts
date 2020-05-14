import MySQLdb
import random
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

    sql="select * from machinelearning_timeseries where series_title='" + this_series + "' and observation_date <= '2019-12-01';"
    #cursor.execute(sql)
    #results=cursor.fetchall()
    try:
        df=pd.read_sql(sql,con=db)
    except:
        print(sql)
        
    # I believe this model as it is configured now allows for the possibility of seasonal adjustments but does not coerce them.
    # This is because seasonal is defaulting to True and D is set to None.
    with StepwiseContext(max_dur=25):
        model=pm.auto_arima(df.inx, start_p=1, start_q=1,
                                test='adf',       # use adftest to find optimal 'd'
                                #test='nm', # for faster performance
                                max_p=3, max_q=3, # maximum p and q
                                m=12,              # frequency of series
                                d=None,           # let model determine 'd'
                                #seasonal=True,   # Seasonality
                                #maxiter=10,
                                start_P=0, 
                                D=None, 
                                #trace=True,
                                #error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True)
    
    model_dictionary=model.to_dict()
    ar_vars=['AR' + str(v+1) for v in range(model_dictionary['order'][0])]
    ma_vars=['MA' + str(v+1) for v in range(model_dictionary['order'][2])]         
    ar_vars_seasonal=['SAR' + str(v+12) for v in range(model_dictionary['seasonal_order'][0])]
    ma_vars_seasonal=['SMA' + str(v+12) for v in range(model_dictionary['seasonal_order'][2])]
    all_vars=ar_vars+ma_vars+ar_vars_seasonal+ma_vars_seasonal
    pvalues=[pv for pv in model_dictionary['pvalues']][1:-1] # the last one is actually the variance and the first is for the const
    coefficients=[c for c in model_dictionary['params']][1:-1] # the last one is actually the variance and the first is for the const
    
    sql=('select id '
         'from machinelearning_series_names '
         'where series_title="' + this_series +'";')
     
    cursor.execute(sql)
    results=cursor.fetchone()   
    series=results[0]
    
    sql="insert into machinelearning_arima_forecast (series_id) values ('" + str(series) + "');"
    cursor.execute(sql)
    db.commit()
    forecast=cursor.lastrowid 
    
    for i in range(len(all_vars)):
        sql='insert into machinelearning_arima_coefficients (variables,coefficients,pvalues,forecasted_series_id) values ("' + str(all_vars[i]) + '",' + str(coefficients[i]) + ',' + str(pvalues[i]) + ',' + str(forecast) + ');'
        cursor.execute(sql)
        db.commit()
 
    n_periods = 12
    predictions, confint=model.predict(n_periods, return_conf_int=True)
    
    sql=[]
    future_dates=['2020-01-01','2020-02-01','2020-03-01','2020-04-01','2020-05-01','2020-06-01','2020-07-01','2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01']
    for j in range(len(predictions)): 
        sql='insert into machinelearning_arima_predictions (future_date,inx,forecasted_series_id,lower_ci,upper_ci) values ("' + str(future_dates[j]) + '",' + str(predictions[j]) + ',' + str(forecast) + ',' + str(confint[j][0]) + ',' + str(confint[j][1]) + ');'
        cursor.execute(sql)
        db.commit()
    
    cursor.close()
    db.close()

if __name__=='__main__':
    
    #forecasts_already_made=arima_predictions.objects.values_list('series_title', flat=True).distinct()
    #selections_list=timeseries.objects.values_list('series_title', flat=True).distinct()#.exclude(series_title__in=forecasts_already_made)
    db=MySQLdb.connect(host="localhost", port=3306, user="root", passwd="a6!modern", db="timeseries")
    cursor=db.cursor()
    
    sql=('select distinct series_title '
         'from machinelearning_timeseries;')
    cursor.execute(sql)
    results=cursor.fetchall()
    selections_list=[]
    for result in results:
        selections_list.append(result[0])
        
    selections_list=random.sample(selections_list, k=10)    
   
    cursor.close()
    db.close()
    
    p=Pool(4)
    preds=p.map(make_predictions, selections_list)

    p.close()
    p.join()