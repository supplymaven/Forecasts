from django.db import models

# Create your models here.

class SandMining(models.Model):
    observation_date=models.DateField() # data is monthly
    ppi=models.DecimalField(max_digits=4, decimal_places=1)