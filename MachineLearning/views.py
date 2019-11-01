import os
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.urls import reverse
from MachineLearning.models import SandMining
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Create your views here.
def selections(request):
    # You can't just arrive at this page; you must first be logged in
    if not request.user.is_authenticated:
        return HttpResponseRedirect(reverse('login')) 
        
    return render(request, '../templates/selections.html', {})

def forecast_model(request):
    # You can't just arrive at this page; you must first be logged in
    if not request.user.is_authenticated:
        return HttpResponseRedirect(reverse('login'))
        
    df=pd.DataFrame.from_records(SandMining.objects.all().values())
    model=pm.auto_arima(df.ppi, start_p=1, start_q=1,
                        test='adf',       # use adftest to find optimal 'd'
                        max_p=5, max_q=5, # maximum p and q
                        m=12,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=True,   # Seasonality
                        start_P=0, 
                        D=None, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)
                      
    #print(model.summary())
    #model.plot_diagnostics(figsize=(20,10))
    #plt.show()
    
    n_periods = 6
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = np.arange(len(df.ppi), len(df.ppi)+n_periods)

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.plot(df.ppi)
    plt.plot(fc_series, color='darkgreen')
    plt.fill_between(lower_series.index, 
                     lower_series, 
                     upper_series, 
                     color='k', alpha=.15)

    plt.title("ARIMA Forecast of Sand Mining PPI")
    #plt.show()
    now=datetime.now()
    timestamp=datetime.timestamp(now)
    image_file_name='arima' + str(int(round(timestamp,0))) + '.png'
    plt.savefig(os.path.join(settings.BASE_DIR, '../Forecasts/MachineLearning/static/images/' + image_file_name))
    predictions=model.predict(n_periods)

    return render(request, '../templates/forecast_model.html', {'image_file_name': image_file_name, 'predictions': [round(p,2) for p in predictions]})              