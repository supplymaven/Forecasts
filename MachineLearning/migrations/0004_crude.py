# Generated by Django 2.2.4 on 2019-12-11 18:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MachineLearning', '0003_auto_20191105_0857'),
    ]

    operations = [
        migrations.CreateModel(
            name='crude',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('wti_real_price', models.DecimalField(decimal_places=4, max_digits=7)),
                ('world_liquid_fuels_production_capacity_change', models.DecimalField(decimal_places=4, max_digits=5)),
                ('avg_num_outstanding_oil_futures_contract', models.DecimalField(decimal_places=4, max_digits=8)),
                ('assets_under_management', models.DecimalField(decimal_places=4, max_digits=7)),
                ('world_gdp_growth', models.DecimalField(decimal_places=4, max_digits=5)),
                ('world_liquid_fuels_consumpton_change', models.DecimalField(decimal_places=4, max_digits=5)),
            ],
        ),
    ]
