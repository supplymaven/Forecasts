from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class sand_mining(models.Model):
    observation_date=models.DateField(unique=True) # data is monthly
    inx=models.DecimalField(max_digits=4, decimal_places=1)
    
class zillow(models.Model):
    observation_date=models.DateField(unique=True) # data is monthly
    inx=models.DecimalField(max_digits=8, decimal_places=6)
    
class nasdaq(models.Model):
    observation_date=models.DateField(unique=True) # data is daily
    inx=models.DecimalField(max_digits=6, decimal_places=2)
    
class yale(models.Model):
    observation_date=models.DateField(unique=True) # data is annual
    inx=models.DecimalField(max_digits=8, decimal_places=6)
    
class sp_ratios(models.Model):
    observation_date=models.DateField(unique=True) # data is monthly
    inx=models.DecimalField(max_digits=3, decimal_places=2)
    
class corporate_bond_yield_rates(models.Model):
    observation_date=models.DateField(unique=True) # data is daily
    inx=models.DecimalField(max_digits=4, decimal_places=2)
    
class commodity_indices(models.Model):
    observation_date=models.DateField(unique=True) # data is daily
    inx=models.DecimalField(max_digits=6, decimal_places=2)
   
class crude(models.Model):
    frequency=models.DateField(unique=True, null=True) # quarterly
    wti_real_price=models.DecimalField(max_digits=7, decimal_places=4) # dependent variable IND.04
    non_opec_liquid_fuels_production=models.DecimalField(max_digits=6, decimal_places=4) #IND.03
    saudi_arabia_crude_oil_production_change=models.DecimalField(max_digits=6, decimal_places=4) # IND.19
    opec_spare_production_capacity=models.DecimalField(max_digits=6, decimal_places=4)  # IND.21
    non_oecd_liquid_fuels_consumption_change=models.DecimalField(max_digits=6, decimal_places=4) # IND.27
    non_oecd_gdp_growth=models.DecimalField(max_digits=10, decimal_places=4) # IND.28
    oecd_liquid_fuels_consumption_change=models.DecimalField(max_digits=6, decimal_places=4) # IND.40
    wti_crude_12_1_futures_price_spread_change=models.DecimalField(max_digits=6, decimal_places=4) # IND.24
    oecd_liquid_fuels_inventories_change=models.DecimalField(max_digits=7, decimal_places=4) # IND.23
    avg_num_outstanding_oil_futures_contract=models.DecimalField(max_digits=8, decimal_places=4) # IND.42
    assets_under_management=models.DecimalField(max_digits=7, decimal_places=4) # IND.53
    world_gdp_growth=models.DecimalField(max_digits=10, decimal_places=4) # IND.06
    world_liquid_fuels_consumption_change=models.DecimalField(max_digits=10, decimal_places=4) # IND.29
    dow_jones_ubs_commodity_index=models.DecimalField(max_digits=7, decimal_places=4) # IND.54
   
# this is to hold the data that Garrett's python programs grab from various apis into a csv file  
# mysql --local-infile=1 -u root -p
# set GLOBAL local_infile=1;
# load data local infile 'C:/users/light/desktop/lightsquaresolutions/gg-supply-maven-master/matt.csv' into table machinelearning_timeseries fields terminated by ',' lines terminated by '\n' ignore 1 lines (@col1,@col2,@col3) set observation_date=CONCAT(SUBSTRING_INDEX(@col2,"/",-1),"-",SUBSTRING_INDEX(@col2,"/",1),"-1"), series_title=@col1, inx=@col3;
class timeseries(models.Model):
    observation_date=models.DateField(unique=False, null=True) # monthly
    series_title=models.CharField(db_index=True, max_length=100, verbose_name="Series Title")
    inx=models.DecimalField(max_digits=10, decimal_places=4)
    
# we store ARIMA predictions for every index here    
class arima_predictions(models.Model):
    future_date=models.DateField(unique=False, null=True)
    series_title=models.CharField(db_index=True, max_length=100, verbose_name="Series Title")
    inx=models.DecimalField(max_digits=10, decimal_places=4)  

# Stored here are the series the user has ran regressions on (useful for breadcrumbs and storing user interests)
class series_visited(models.Model):    
    date_time_clicked=models.DateTimeField()
    user=models.ForeignKey(User, verbose_name="User", on_delete=models.CASCADE)
    series=models.CharField(max_length=100, verbose_name="Series Title")