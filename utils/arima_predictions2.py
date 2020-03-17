import os
import django
import sys
from django.conf import settings
from django.db.models import Max
from contextlib import closing
from django.db import connection

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Forecasts.settings')
django.setup()



from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pandas as pd
import numpy as np
from dateutil.relativedelta import *
from MachineLearning.models import timeseries, arima_predictions

def sql_simple_insert_executemany(n_records):
    with closing(connection.cursor()) as cursor:
        cursor.executemany(
            'INSERT INTO machinelearning_arima_predictions (future_date, series_title, inx)'
            'VALUES (%s, %s, %s)',
            [(i, str(i), timezone.now()) for i in xrange(0, n_records)],
        )


if __name__=='__main__':
    #f=open("arima_predictions.csv", "w")
    forecasts_already_made=arima_predictions.objects.values_list('series_title', flat=True).distinct()
    selections_list=timeseries.objects.values_list('series_title', flat=True).distinct().exclude(series_title__in=forecasts_already_made)

    for this_series in selections_list:
        df=pd.DataFrame.from_records(timeseries.objects.filter(series_title=this_series).values())
        max_date=timeseries.objects.filter(series_title=this_series).aggregate(Max('observation_date'))
        # I believe this model as it is configured now allows for the possibility of seasonal adjustments but does not coerce them.
        # This is because seasonal is defaulting to True and D is set to None.
        model=pm.auto_arima(df.inx, start_p=1, start_q=1,
                                    test='adf',       # use adftest to find optimal 'd'
                                    max_p=5, max_q=5, # maximum p and q
                                    m=12,              # frequency of series
                                    d=None,           # let model determine 'd'
                                    #seasonal=True,   # Seasonality
                                    start_P=0, 
                                    D=None, 
                                    #trace=True,
                                    #error_action='ignore',  
                                    suppress_warnings=True, 
                                    stepwise=True)
                                    
        #model_dictionary=model.to_dict()
        #coefficients=model.params()                    
                      
        n_periods = 12

        #fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
        #index_of_fc = np.arange(len(df.inx), len(df.inx)+n_periods)
        
        predictions=model.predict(n_periods)
        month_increment=1
        for prediction in predictions:         
            new_prediction=arima_predictions(future_date=max_date['observation_date__max']+relativedelta(months=+month_increment), series_title=this_series, inx=prediction)
            new_prediction.save()
            #f.write(str(max_date['observation_date__max']+relativedelta(months=+month_increment)) + ',' + this_series + ',' + str(prediction) + '\n')
            #month_increment+=1   
            
    #f.close()        