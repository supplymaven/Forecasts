# Generated by Django 2.2.4 on 2020-02-25 15:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MachineLearning', '0014_auto_20200124_1226'),
    ]

    operations = [
        migrations.AlterField(
            model_name='timeseries',
            name='series_title',
            field=models.CharField(db_index=True, max_length=100, verbose_name='Series Title'),
        ),
    ]
