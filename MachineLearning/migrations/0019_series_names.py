# Generated by Django 2.2.4 on 2020-04-11 20:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MachineLearning', '0018_auto_20200408_0709'),
    ]

    operations = [
        migrations.CreateModel(
            name='series_names',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('series_title', models.CharField(db_index=True, max_length=100, verbose_name='Series Title')),
            ],
        ),
    ]