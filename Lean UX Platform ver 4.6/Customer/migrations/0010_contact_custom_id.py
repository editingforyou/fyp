# Generated by Django 4.2.3 on 2023-10-15 14:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Customer', '0009_remove_contact_urls_contact_url_delete_url'),
    ]

    operations = [
        migrations.AddField(
            model_name='contact',
            name='custom_id',
            field=models.CharField(default='some_default_value', max_length=20, unique=True),
        ),
    ]