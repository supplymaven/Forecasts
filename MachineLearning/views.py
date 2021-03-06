import os
import math
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.urls import reverse
from MachineLearning.models import sand_mining, zillow, nasdaq, yale, sp_ratios, corporate_bond_yield_rates, commodity_indices, crude, timeseries, arima_predictions, series_visited, series_names, econometric_predictions, econometric_coefficients, econometric_forecast, arima_predictions, arima_coefficients, arima_forecast
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import backend as K
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from utils.lstm_utils import *
from utils.genetic_algorithm import *

def home(request):
    # You can't just arrive at this page; you must first be logged in
    if not request.user.is_authenticated:
        return HttpResponseRedirect(reverse('login'))

    selections_list=series_names.objects.filter(id__in=arima_forecast.objects.values_list('series', flat=True).distinct()).values_list('series_title',flat=True).distinct()
    
    if request.POST:
    
        df=pd.DataFrame.from_records(timeseries.objects.filter(series_title=request.POST['timeseries']).values())[-6:]
        actual_periods=6
        n_periods=6
        series_name=series_names.objects.get(series_title=request.POST['timeseries'])
        forecast=arima_forecast.objects.get(series=series_name)
        prediction_objects=arima_predictions.objects.filter(forecasted_series=forecast).order_by('future_date')
        coefficient_objects=arima_coefficients.objects.filter(forecasted_series=forecast)
        
        predictions=[]
        lower_cis=[]
        upper_cis=[]
        for p in prediction_objects:
            predictions.append(p.inx)
            lower_cis.append(p.lower_ci)
            upper_cis.append(p.upper_ci)
            
        coefficients=[]
        variables=[]
        pvalues=[]
        for c in coefficient_objects:
            coefficients.append(c.coefficients)
            variables.append(c.variables)
            pvalues.append(c.pvalues)
            
   
    
        # make series for plotting purpose
        index_of_fc = np.arange(actual_periods, actual_periods+n_periods)
        fc_series = pd.Series(np.asarray(predictions[-6:]), index=index_of_fc)
        lower_series = pd.Series(np.asarray(lower_cis[-6:]), index=index_of_fc)
        upper_series = pd.Series(np.asarray(upper_cis[-6:]), index=index_of_fc)

        # Plot
        formatted_series=request.POST['timeseries']
        df.index=np.arange(actual_periods)
        plt.plot(df.inx)
        plt.plot(fc_series, color='darkgreen')
        print(lower_series.index)
        print(lower_series)
        print(upper_series)
        plt.fill_between(lower_series.index, 
                        lower_series.astype(float), 
                        upper_series.astype(float), 
                        color='k', alpha=.15)

        plt.title("ARIMA Forecast")
        #print([fc])
        #plt.table(cellText=[['' for i in range(len(df.inx))]+[round(f,1) for f in fc]])
        #plt.show()
        now=datetime.now()
        timestamp=datetime.timestamp(now)
        image_file_name='arima' + str(int(round(timestamp,0))) + '.png'
        plt.savefig(os.path.join(settings.BASE_DIR, str(settings.STATIC_ROOT) + '/images/' + image_file_name))
        # clear the figure
        plt.clf()
        
        df=pd.DataFrame(list(arima_coefficients.objects.filter(forecasted_series=forecast).values('variables','coefficients','pvalues')))
        future_dates=['2020-01-01','2020-02-01','2020-03-01','2020-04-01','2020-05-01','2020-06-01','2020-07-01','2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01']
        
        predictions=zip(predictions,future_dates)

        return render(request, '../templates/home_arima.html', {'selections_list': selections_list, 'image_file_name': image_file_name, 'coefficients': coefficients, 'pvalues': pvalues, 'variables': variables, 'predictions': predictions, 'lower_cis': lower_cis, 'upper_cis': upper_cis, 'timeseries': request.POST['timeseries'],'df': df, })
    else:    
        return render(request, '../templates/home.html', {'selections_list': selections_list})    

# deprecated
def home_regression(request):
    # the arima predictions for these indexes did not turn out right so we exclude them for the time being
    to_exclude=['Dow Jones Industrial Average','Gold','PPI Commodity data for Machinery and equipment-Parts and attachments for industrial process furnaces','PPI Commodity data for Nonmetallic mineral products-Handmade pressed and blown glassware not seasona','PPI Commodity data for Textile products and apparel-Womens and girls tailored jackets and vests ex','PPI Commodity data for Transportation equipment-Air brake and other brake equipment not seasonally a','PPI industry data for Industrial process furnace and oven mfg-Parts and attachments for industrial p','PPI industry data for Mechanical power transmission equipment mfg-Plain bearings and bushings not se','PPI industry data for Nonferrous metal (except copper and aluminum) rolling drawing and extruding-Al','PPI industry data for Other metal container mfg-Steel shipping barrels & drums exc. beer barrels (mo','PPI industry data for Power boiler and heat exchanger mfg-Fabricated heat exchangers and steam conde','PPI industry data for Railroad rolling stock mfg-Air brake and other brake equipment not seasonally','PPI industry data for Sawmills-Secondary products not seasonally adjusted','S&P 500','Utility (piped) gas per therm in Atlanta-Sandy Springs-Roswell GA average price not seasonally adjus','Utility (piped) gas service in Atlanta-Sandy Springs-Roswell GA all urban consumers not seasonally a','WTI Crude Oil']
    selections_list=series_names.objects.filter(id__in=econometric_forecast.objects.values_list('series', flat=True).distinct()).values_list('series_title',flat=True).distinct().exclude(series_title__in=to_exclude)
    if request.POST:
        series=series_names.objects.get(series_title=request.POST['timeseries'])
        forecasted_series=econometric_forecast.objects.get(series=series)
        # Here we create our breadcrumbs by saving what regression was just performed and retrieving all that was performed within the last hour
        current_time=datetime.now()
        visited=series_visited(date_time_clicked=current_time, user=request.user, series=request.POST['timeseries'])
        visited.save()
        current_time=datetime.now()
        one_hour_ago=current_time-timedelta(hours=1)
        breadcrumbs=series_visited.objects.filter(user=request.user, date_time_clicked__range=(one_hour_ago, current_time)).order_by('date_time_clicked')
        breadcrumb_series=[]
        # we want a unique list of series chronologically ordered (this is our breadcrumb trail)
        for breadcrumb in breadcrumbs:
            if breadcrumb.series not in breadcrumb_series:
                breadcrumb_series.append(breadcrumb.series)
                
        # don't send anything for breadcrumbs if there is only one series to report
        if len(breadcrumb_series)==1:
            breadcrumb_series=[]
             
        now=datetime.now()
        timestamp=datetime.timestamp(now)
        plt.title("Econometric Forecast of " + request.POST['timeseries'])
        
        forecasted_series=econometric_forecast.objects.get(series=series)
        indep_vars=econometric_coefficients.objects.filter(forecasted_series=forecasted_series).values_list('independent_variables',flat=True)
        coefs=econometric_coefficients.objects.filter(forecasted_series=forecasted_series).values_list('coefficients',flat=True)
        p_values=econometric_coefficients.objects.filter(forecasted_series=forecasted_series).values_list('pvalues',flat=True)
        df=pd.DataFrame(list(econometric_coefficients.objects.filter(forecasted_series=forecasted_series).values('independent_variables','coefficients','pvalues')))
        predictions_future=econometric_predictions.objects.filter(forecasted_series=forecasted_series).values_list('inx',flat=True)
        future_dates=['2020-01-01','2020-02-01','2020-03-01','2020-04-01','2020-05-01','2020-06-01','2020-07-01','2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01']
        
        predictions=zip(predictions_future,future_dates)
        
        # plotting
        dts=['07-19','08-19','09-19','10-19','11-19','12-19','01-20','02-20','03-20','04-20','05-20','06-20']
        actuals=np.array(list(timeseries.objects.filter(series_title=request.POST['timeseries'], observation_date__in=['2019-07-01','2019-08-01','2019-09-01','2019-10-01','2019-11-01','2019-12-01']).order_by('observation_date').values_list('inx',flat=True)))
        preds=np.array(predictions_future[:6])

        # clear the figure
        plt.clf()
        plt.plot(dts[:6],actuals.astype(float))
        plt.plot(dts[6:],preds.astype(float))
        plt.margins(0.01)
        image_file_name='econometric' + str(int(round(timestamp,0))) + '.png'
        plt.savefig(os.path.join(settings.BASE_DIR, str(settings.STATIC_ROOT) + '/images/' + image_file_name))
        
        # clear the figure
        plt.clf()
        
        return render(request, '../templates/home.html', {'image_file_name': image_file_name, 'predictions': predictions, 'timeseries': request.POST['timeseries'], 'selections_list': selections_list, 'breadcrumbs': breadcrumb_series, 'df': df,})
    else:    
        return render(request, '../templates/home.html', {'selections_list': selections_list})

# Example of template
def index(request):
    return render(request, '../templates/index.html', {})

# Create your views here.
def selections(request):
    # You can't just arrive at this page; you must first be logged in
    if not request.user.is_authenticated:
        return HttpResponseRedirect(reverse('login')) 
    #selections=timeseries.objects.all()
    selections_list=timeseries.objects.values_list('series_title', flat=True).distinct()

    return render(request, '../templates/selections.html', {'selections_list': selections_list})
    
def arima(request):
    # machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
    # You can't just arrive at this page; you must first be logged in
    if not request.user.is_authenticated:
        return HttpResponseRedirect(reverse('login'))

    if request.POST:
    
        selections_list=timeseries.objects.values_list('series_title', flat=True).distinct()
    
        df=pd.DataFrame.from_records(timeseries.objects.filter(series_title=request.POST['timeseries']).values())
        m=12
        formatted_series='PPI'
    
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
        plt.savefig(os.path.join(settings.BASE_DIR, str(settings.STATIC_ROOT) + '/images/' + image_file_name))
        # clear the figure
        plt.clf()
        predictions=model.predict(n_periods)
        #print(model.params())
        #print(model.pvalues())
        #print(model.summary())
        #print(model.to_dict())
        
        pvalues=model.pvalues()
        ar_vars=['AR' + str(v+1) for v in range(model_dictionary['order'][0])]
        ma_vars=['MA' + str(v+1) for v in range(model_dictionary['order'][2])]
    
        return render(request, '../templates/forecast_model.html', {'selections_list': selections_list, 'image_file_name': image_file_name, 'coefficients': coefficients, 'pvalues': pvalues, 'ar_vars': ar_vars, 'ma_vars': ma_vars, 'predictions': [round(p,2) for p in predictions]})

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
        plt.savefig(os.path.join(settings.BASE_DIR, str(settings.STATIC_ROOT) + '/images/' + image_file_name))
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
        # define input sequence
        raw_seq=[float(i) for i in df['inx'].values.tolist()]
        # choose a number of time steps
        n_steps = 3
        # split into samples
        X, y = split_sequence(raw_seq, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        # define model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=200, verbose=0)
        # demonstrate prediction  
        # the last 2 of the X sequence and the last y of the y sequence
        x_input=np.array([i[0] for i in X[-1][-2:]]+[y[-1]])
        
        # We only take the last three elements of a series to make a prediction of the next element.
        # We continually append the input array with predictions, so the input array continually grows.
        # However, we use [-3:] to make sure we are always only using the last 3 elements for the next
        # prediction.
        for i in range(0,12):
            x_input_reshaped = x_input[-3:].reshape((1, n_steps, n_features))
            yhat = model.predict(x_input_reshaped, verbose=0)
            x_input=np.append(x_input, yhat)

        K.clear_session()
        plt.plot(raw_seq)
        plt.plot(pd.Series(x_input.tolist()[3:],index=np.arange(len(raw_seq),len(raw_seq)+len(x_input.tolist()[3:]))), color='darkgreen')
        plt.title("LSTM Forecast of " + formatted_series)
        now=datetime.now()
        timestamp=datetime.timestamp(now)
        image_file_name='lstm' + str(int(round(timestamp,0))) + '.png'
        plt.savefig(os.path.join(settings.BASE_DIR, str(settings.STATIC_ROOT) + '/images/' + image_file_name))
        # clear the figure
        plt.clf()
        return render(request, '../templates/forecast_model.html', {'image_file_name': image_file_name, 'predictions': [round(p,2) for p in x_input[3:]]})  
    elif model=='econometric':
        df=pd.DataFrame(list(crude.objects.all().values()))
        target=pd.DataFrame(list(crude.objects.all().values('wti_real_price')))
        # use a genetic algorithm to select the independent (explanatory) variables
        num_generations=20
        num_variables=13 # for crude
        size_of_chromosome_population=100 # 2^5
        crossover_probability=0.7
        mutation_probability=0.001
        #                                  num_possble_vars, num_possible_combinations_of_vars
        chromosome_population=generate_initial_population(num_variables,size_of_chromosome_population)
        min_rmse_test=math.inf
        most_fit_chromosome=''
        for i in range(num_generations):
        
            new_chromosome_population=[]
           
            while len(new_chromosome_population)!=len(chromosome_population):
                # get fitness of each chromosome
                chromosome_fitness_pairs=[]
                accumulated_fitness=0
                for chromosome in chromosome_population:
                    # get all indices in bit string where bit string is a '1'
                    # we don't want the first three colums, however: id and frequency and the dependent variable
                    indices=[i+3 for i, x in enumerate(chromosome) if x == '1']
                    # these are the dataframe columns corresponding to those bits in the bit string that are equal to 1
                    X=df.iloc[:,indices]        
                    X=sm.add_constant(X)
                    y=target["wti_real_price"]
                    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
                    #X_train=sm.add_constant(X_train)
                    model=sm.OLS(y_train.astype(float), X_train.astype(float)).fit()
                    predictions_train=model.predict(X_train.astype(float))
                    predictions_test=model.predict(X_test.astype(float))
                    rmse_train=round(rmse(y_train.astype(float),predictions_train.astype(float)),3)
                    # We will use this as our fitness function in the GA
                    rmse_test=round(rmse(y_test.astype(float),predictions_test.astype(float)),10) 
                    if rmse_test<min_rmse_test:
                        min_rmse_test=rmse_test
                        most_fit_chromosome=chromosome
                    chromosome_fitness_pairs.append((chromosome,1/rmse_test))
                    accumulated_fitness+=1/rmse_test
                    
                # get the fitness ratios    
                chromosome_fitness_pairs=[(i,j/accumulated_fitness) for (i,j) in chromosome_fitness_pairs]    
                
                mating_pair=select_chromosome_pair(chromosome_fitness_pairs)
                new_pair=mate_pair(mating_pair, crossover_probability, mutation_probability)
                new_chromosome_population.append(new_pair[0])
                new_chromosome_population.append(new_pair[1])
            chromosome_population=[new_chromosome for new_chromosome in new_chromosome_population]
            new_chromosome_population=[]
            
        #print(chromosome_population)
        #print(most_fit_chromosome)
        #print(min_rmse_test)       
        #X=df[["non_opec_liquid_fuels_production","opec_spare_production_capacity","non_oecd_liquid_fuels_consumption_change","non_oecd_gdp_growth","avg_num_outstanding_oil_futures_contract","world_liquid_fuels_consumption_change"]]
        # get all indices in bit string where bit string is a '1'
        # we don't want the first three colums, however: id and frequency and the dependent variable
        indices=[i+3 for i, x in enumerate(most_fit_chromosome) if x == '1']
        # these are the dataframe columns corresponding to those bits in the bit string that are equal to 1
        X=df.iloc[:,indices]        
        X=sm.add_constant(X)
        y=target["wti_real_price"]
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        model=sm.OLS(y_train.astype(float), X_train.astype(float)).fit()
        predictions_train=model.predict(X_train.astype(float))
        predictions_test=model.predict(X_test.astype(float))
        rmse_train=round(rmse(y_train.astype(float),predictions_train.astype(float)),3)
        # We will use this as our fitness function in the GA
        rmse_test=round(rmse(y_test.astype(float),predictions_test.astype(float)),3) 
        now=datetime.now()
        timestamp=datetime.timestamp(now)
        plt.title("Econometric Forecast of " + series.title())
        plt.plot(y_train.astype(float),label='y_train_actual')
        plt.plot(y_test.astype(float),label='y_test_actual')
        plt.plot(predictions_train.astype(float),label='y_train_pred')
        plt.plot(predictions_test.astype(float),label='y_test_pred')
        plt.legend(loc='upper right')
        image_file_name='econometric' + str(int(round(timestamp,0))) + '.png'
        plt.savefig(os.path.join(settings.BASE_DIR, str(settings.STATIC_ROOT) + '/images/' + image_file_name))
        # clear the figure
        plt.clf()
        summ=model.summary()
        
        # standardize data to get standardized coefficients for donut graph
        X=X.drop(columns=['const']) # X had a column vector of 1s in the original dataframe so we remove it prior to normalizing and adding a constant to the new model
        X_standardized=(X.astype(float)-X.astype(float).mean())/X.astype(float).std()
        X_standardized=sm.add_constant(X_standardized)
        y_standardized=(y.astype(float)-y.astype(float).mean())/y.astype(float).std()
        
        X_train_std, X_test_std, y_train_std, y_test_std=train_test_split(X_standardized, y_standardized, test_size=0.2, random_state=42, shuffle=False)
        model_standardized=sm.OLS(y_train_std.astype(float), X_train_std.astype(float)).fit()
        coefficients=dict(model_standardized.params)
        
        # donut chart of normalized coefficients
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(aspect="equal"))
        ax.axis("off")
        ax = fig.add_subplot(211)
        coeffs=[key for key in coefficients.keys() if key!='const']
        data=[abs(coefficients[key]) for key in coefficients.keys() if key!='const']
        wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40, radius=1.25)
        ax2 = fig.add_subplot(212)
        ax2.axis("off") 
        ax2.legend(wedges,coeffs, loc="center")
        #bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        #kw = dict(arrowprops=dict(arrowstyle="-"),
        #          bbox=bbox_props, zorder=0, va="center")

        """
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(coeffs[i].replace('_','\n'), xy=(x, y), xytext=(1.05*np.sign(x), 1.1*y),
                        horizontalalignment=horizontalalignment, **kw)
        """
        ax.set_title("Impact of Coefficients")

        image_file_name_donut='coefficients' + str(int(round(timestamp,0))) + '.png'
        plt.savefig(os.path.join(settings.BASE_DIR, str(settings.STATIC_ROOT) + '/images/' + image_file_name_donut))
        
        plt.clf()
        
        return render(request, '../templates/econometric_model.html', {'summary1':summ.tables[0].as_html(), 'summary2': summ.tables[1].as_html(), 'summary3': summ.tables[2].as_html(), 'image_file_name': image_file_name, 'image_file_name_donut': image_file_name_donut, 'rmse_train': rmse_train, 'rmse_test': rmse_test })
        