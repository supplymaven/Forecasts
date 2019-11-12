import os
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.urls import reverse
from MachineLearning.models import sand_mining, zillow, nasdaq, yale, sp_ratios, corporate_bond_yield_rates, commodity_indices
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from utils.lstm_utils import *


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
        # transform data to be stationary
        raw_values=pd.to_numeric(df['inx']).to_numpy()
        print(raw_values)
        diff_values=difference(raw_values,1)
        #print(diff_values)
        # transform data to be supervised learning
        supervised=timeseries_to_supervised(diff_values, 1)
        supervised_values=supervised.values
        #print(supervised_values)
        # split data into train and test-sets
        train, test = supervised_values[0:-12], supervised_values[-12:]
        
        # transform the scale of the data
        scaler, train_scaled, test_scaled = scale(train, test)
 
        # fit the model
        lstm_model = fit_lstm(train_scaled, 1, 1, 4)
        # forecast the entire training dataset to build up state for forecasting
        train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
        lstm_model.predict(train_reshaped, batch_size=1)
         
        # walk-forward validation on the test data
        predictions = list()
        for i in range(len(test_scaled)):
            # make one-step forecast
            X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
            yhat = forecast_lstm(lstm_model, 1, X)
            # invert scaling
            yhat = invert_scale(scaler, X, yhat)
            # invert differencing
            yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
            # store forecast
            predictions.append(yhat)
            expected = raw_values[len(train) + i + 1]
            print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
         
        # report performance
        rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
        print('Test RMSE: %.3f' % rmse)
        # line plot of observed vs predicted
        plt.plot(raw_values[-12:])
        plt.plot(predictions)
        plt.show()
                
                
        
        return HttpResponse("")