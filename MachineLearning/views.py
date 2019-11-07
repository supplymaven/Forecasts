import os
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.urls import reverse
from MachineLearning.models import sand_mining, zillow, nasdaq, yale, sp_ratios, corporate_bond_yield_rates, commodity_indices
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from utils import lstm_utils

# Create your views here.
def selections(request):
    # You can't just arrive at this page; you must first be logged in
    if not request.user.is_authenticated:
        return HttpResponseRedirect(reverse('login')) 
        
    return render(request, '../templates/selections.html', {})

def forecast_model(request, series, model):
    # You can't just arrive at this page; you must first be logged in
    if not request.user.is_authenticated:
        return HttpResponseRedirect(reverse('login'))
    
    m=12
    formatted_series=''
    if series=='sand-mining':
        df=pd.DataFrame.from_records(sand_mining.objects.all().values())
        m=12
        formatted_series='Sand Mining'
    elif series=='zillow':
        df=pd.DataFrame.from_records(zillow.objects.all().values())
        m=12
        formatted_series='Zillow'
    elif series=='nasdaq':
        df=pd.DataFrame.from_records(nasdaq.objects.all().values())
        m=365
        formatted_series='NASDAQ'
    elif series=='yale':
        df=pd.DataFrame.from_records(yale.objects.all().values())
        m=1
        formatted_series='Yale'
    elif series=='sp-ratios':
        df=pd.DataFrame.from_records(sp_ratios.objects.all().values())
        m=12
        formatted_series='S&P 500 Ratios'
    elif series=='corporate-bond-yield-rates':
        df=pd.DataFrame.from_records(corporate_bond_yield_rates.objects.all().values())
        m=365
        formatted_series='Corporate Bond Yield Rates'
    elif series=='commodity-indices':
        df=pd.DataFrame.from_records(commodity_indices.objects.all().values())
        m=365
        formatted_series='Commodity Indices'
        
    if model=='arima':
        
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
                      
        n_periods = 12
        fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
        index_of_fc = np.arange(len(df.inx), len(df.inx)+n_periods)

        # make series for plotting purpose
        fc_series = pd.Series(fc, index=index_of_fc)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)

        # Plot
        plt.plot(df.inx)
        plt.plot(fc_series, color='darkgreen')
        plt.fill_between(lower_series.index, 
                        lower_series, 
                        upper_series, 
                        color='k', alpha=.15)

        plt.title("ARIMA Forecast of " + formatted_series)
        #print([fc])
        #plt.table(cellText=[['' for i in range(len(df.inx))]+[round(f,1) for f in fc]])
        #plt.show()
        now=datetime.now()
        timestamp=datetime.timestamp(now)
        image_file_name='arima' + str(int(round(timestamp,0))) + '.png'
        plt.savefig(os.path.join(settings.BASE_DIR, '../Forecasts/MachineLearning/static/images/' + image_file_name))
        # clear the figure
        plt.clf()
        predictions=model.predict(n_periods)
        #print(model.params())
        #print(model.pvalues())
        print(model.summary())
        print(model.to_dict())
        model_dictionary=model.to_dict()
        coefficients=model.params()
        pvalues=model.pvalues()
        ar_vars=['AR' + str(v+1) for v in range(model_dictionary['order'][0])]
        ma_vars=['MA' + str(v+1) for v in range(model_dictionary['order'][2])]
    
        return render(request, '../templates/forecast_model.html', {'image_file_name': image_file_name, 'coefficients': coefficients, 'pvalues': pvalues, 'ar_vars': ar_vars, 'ma_vars': ma_vars, 'predictions': [round(p,2) for p in predictions]})              
    elif model=='lstm':
        #supervised=lstm_utils.timeseries_to_supervised(df['inx'], 1)
        #print(supervised)
        print(df['inx'])
        diff=lstm_utils.difference(df['inx'],1)
        #print(diff)
        inverted = list()
        print(len(diff))
        print(len(df['inx']))
        for i in range(len(diff)):
            value = lstm_utils.inverse_difference(df['inx'].tolist(), diff[i], len(df['inx'])-i)
            inverted.append(value)
        inverted = pd.Series(inverted)
        print(inverted)
        
        return HttpResponse("")