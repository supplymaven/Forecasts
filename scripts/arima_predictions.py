

from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pandas as pd
import numpy as np
from MachineLearning.models import timeseries

if __name__=='__main__':
    df=pd.DataFrame.from_records(timeseries.objects.filter(series_title="Gold").values())
    model=pm.auto_arima(df.inx, start_p=1, start_q=1,
                                test='adf',       # use adftest to find optimal 'd'
                                max_p=5, max_q=5, # maximum p and q
                                m=m,              # frequency of series
                                d=None,           # let model determine 'd'
                                seasonal=False,   # Seasonality
                                start_P=0, 
                                D=None, 
                                trace=True,
                                error_action='ignore',  
                                suppress_warnings=True, 
                                stepwise=True)
                                
    model_dictionary=model.to_dict()
    coefficients=model.params()                    
                  
    n_periods = 12

    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = np.arange(len(df.inx), len(df.inx)+n_periods)