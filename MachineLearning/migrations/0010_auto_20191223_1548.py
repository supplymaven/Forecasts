# Generated by Django 2.2.4 on 2019-12-23 21:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MachineLearning', '0009_auto_20191223_1547'),
    ]

    operations = [
        migrations.AlterField(
            model_name='crude',
            name='non_oecd_gdp_growth',
            field=models.DecimalField(decimal_places=4, max_digits=10),
        ),
    ]