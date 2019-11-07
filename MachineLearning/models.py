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
    
    
    
    