from django.db import models
from django_countries.fields import CountryField

class Contact(models.Model):
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    email = models.EmailField()
    subject = models.CharField(max_length=255)
    message = models.TextField()
    company = models.CharField(max_length=255, blank=True, null=True)
    country = CountryField(blank_label='(select country)')

    def __str__(self):
        return self.email
