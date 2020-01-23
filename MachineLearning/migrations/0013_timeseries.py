# Generated by Django 2.2.4 on 2020-01-23 18:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MachineLearning', '0012_auto_20191223_1552'),
    ]

    operations = [
        migrations.CreateModel(
            name='timeseries',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('frequency', models.DateField(null=True)),
                ('series_title', models.CharField(max_length=100, verbose_name='Series Title')),
                ('price', models.DecimalField(decimal_places=4, max_digits=10)),
            ],
        ),
    ]