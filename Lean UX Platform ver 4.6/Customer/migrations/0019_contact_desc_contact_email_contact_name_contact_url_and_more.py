# Generated by Django 4.2.3 on 2023-10-15 14:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Customer', '0018_remove_contact_desc_remove_contact_email_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='contact',
            name='desc',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='contact',
            name='email',
            field=models.EmailField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='contact',
            name='name',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='contact',
            name='url',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='contact',
            name='webName',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]