# Generated by Django 2.2.4 on 2019-12-11 18:48

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('MachineLearning', '0004_crude'),
    ]

    operations = [
        migrations.RenameField(
            model_name='crude',
            old_name='world_liquid_fuels_consumpton_change',
            new_name='world_liquid_fuels_consumption_change',
        ),
    ]
