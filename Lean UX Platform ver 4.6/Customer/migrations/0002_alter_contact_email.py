# Generated by Django 4.2.3 on 2023-07-08 17:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Customer', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='contact',
            name='email',
            field=models.EmailField(max_length=255),
        ),
    ]
