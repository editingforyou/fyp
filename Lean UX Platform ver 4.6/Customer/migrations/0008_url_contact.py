# Generated by Django 4.2.3 on 2023-10-15 11:48

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('Customer', '0007_rename_url_contact_urls'),
    ]

    operations = [
        migrations.AddField(
            model_name='url',
            name='contact',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='Customer.contact'),
        ),
    ]