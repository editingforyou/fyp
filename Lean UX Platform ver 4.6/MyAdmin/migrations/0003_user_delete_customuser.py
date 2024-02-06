# Generated by Django 4.2.3 on 2023-10-17 15:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MyAdmin', '0002_customuser_delete_customerdata'),
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=100)),
                ('first_name', models.CharField(max_length=100)),
                ('last_name', models.CharField(max_length=100)),
                ('email', models.EmailField(max_length=254)),
                ('password', models.CharField(max_length=100)),
                ('jobPosition', models.CharField(max_length=100)),
            ],
        ),
        migrations.DeleteModel(
            name='CustomUser',
        ),
    ]