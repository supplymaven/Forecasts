import os
import django
import sys
import MySQLdb
from django.conf import settings
from django.db.models import Max
from django import db

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
from MachineLearning.models import timeseries, arima_predictions
from multiprocessing import Pool, Process

def make_predictions(this_series):
    df=pd.DataFrame.from_records(timeseries.objects.filter(series_title=this_series).values())
    max_date=timeseries.objects.filter(series_title=this_series).aggregate(Max('observation_date'))
    # I believe this model as it is configured now allows for the possibility of seasonal adjustments but does not coerce them.
    # This is because seasonal is defaulting to True and D is set to None.
    with StepwiseContext(max_dur=5):
        model=pm.auto_arima(df.inx, start_p=1, start_q=1,
                                test='adf',       # use adftest to find optimal 'd'
                                #test='nm', # for faster performance
                                max_p=3, max_q=3, # maximum p and q
                                m=12,              # frequency of series
                                d=None,           # let model determine 'd'
                                #seasonal=True,   # Seasonality
                                maxiter=10,
                                start_P=0, 
                                D=None, 
                                #trace=True,
                                #error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True)
    
    model_dictionary=model.to_dict()
    ar_vars=['AR' + str(v+1) for v in range(model_dictionary['order'][0])]
    ma_vars=['MA' + str(v+1) for v in range(model_dictionary['order'][2])]         
    
    print(model.summary())
    for k,v in model_dictionary.items():
        if k!='resid':
            print(k,v)
            
    #print(ar_vars)
    #print(ma_vars)
    n_periods = 12
    predictions=model.predict(n_periods)
    #print(model.summary())
    month_increment=1
    #str(max_date['observation_date__max']+relativedelta(months=+month_increment)) + ',' + 
    #prediction_inserts=this_series + ',' +','.join([str(p) for p in predictions.tolist()])+'\n'
    #print(prediction_inserts)
    #new_predictions=[]
    sql=[]
    for prediction in predictions: 
        #print(prediction)
        #new_predictions.append(arima_predictions(future_date=max_date['observation_date__max']+relativedelta(months=+month_increment), series_title=this_series, inx=prediction))
        #sql="INSERT INTO MachineLearning_arima_predictions (future_date, series_title, inx) VALUES ('" + str(max_date['observation_date__max']+relativedelta(months=+month_increment)) + "','" + this_series + "'," + str(prediction) + ");"
        #cursor.execute(sql)
        sql.append("'" + str(max_date['observation_date__max']+relativedelta(months=+month_increment)) + "','" + this_series + "'," + str(prediction))
        month_increment+=1
        #new_prediction.save()
    #arima_predictions.objects.bulk_create(new_predictions)
    #db.reset_queries() # prevents memory leak
    #    prediction_inserts+=str(max_date['observation_date__max']+relativedelta(months=+month_increment)) + ',' + this_series + ',' + str(prediction) + '\n'
    
    #return prediction_inserts
    #db.commit()
    return sql
if __name__=='__main__':
    
    #forecasts_already_made=arima_predictions.objects.values_list('series_title', flat=True).distinct()
    selections_list=timeseries.objects.values_list('series_title', flat=True).distinct()#.exclude(series_title__in=forecasts_already_made)
    p=Pool(1)
    preds=p.map(make_predictions, selections_list[:5])
    #preds=p.imap_unordered(make_predictions, selections_list, 500)
    p.close()
    p.join()
    f=open("arima_predictions.txt", "w")
    for pred in preds:
        for pr in pred:
            f.write(str(pr)+'\n')
    f.close()
    
    #cursor.close()
    #db.close()
    