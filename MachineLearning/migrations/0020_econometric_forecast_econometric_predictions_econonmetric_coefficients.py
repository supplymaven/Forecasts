# Generated by Django 2.2.4 on 2020-04-18 11:37

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('MachineLearning', '0019_series_names'),
    ]

    operations = [
        migrations.CreateModel(
            name='econometric_forecast',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('series', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='MachineLearning.series_names', verbose_name='Series')),
            ],
        ),
        migrations.CreateModel(
            name='econonmetric_coefficients',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('coefficients', models.DecimalField(decimal_places=4, max_digits=10)),
                ('pvalues', models.DecimalField(decimal_places=4, max_digits=10)),
                ('forecasted_series', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='MachineLearning.econometric_forecast')),
            ],
        ),
        migrations.CreateModel(
            name='econometric_predictions',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('future_date', models.DateField(null=True)),
                ('inx', models.DecimalField(decimal_places=4, max_digits=10)),
                ('forecasted_series', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='MachineLearning.econometric_forecast')),
            ],
        ),
    ]
