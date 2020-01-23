from django.db import models

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
class timeseries(models.Model):
    frequency=models.DateField(unique=False, null=True) # monthly
    series_title=models.CharField(max_length=100, verbose_name="Series Title")
    price=models.DecimalField(max_digits=10, decimal_places=4)