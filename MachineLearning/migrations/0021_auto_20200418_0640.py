# Generated by Django 2.2.4 on 2020-04-18 11:40

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('MachineLearning', '0020_econometric_forecast_econometric_predictions_econonmetric_coefficients'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='econonmetric_coefficients',
            new_name='econometric_coefficients',
        ),
    ]
