# Generated by Django 2.2.4 on 2019-12-23 21:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MachineLearning', '0011_auto_20191223_1551'),
    ]

    operations = [
        migrations.AlterField(
            model_name='crude',
            name='world_liquid_fuels_consumption_change',
            field=models.DecimalField(decimal_places=4, max_digits=10),
        ),
    ]