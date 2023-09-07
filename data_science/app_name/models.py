from django.db import models

# Create your models here.

class Overtime(models.Model):
    name = models.CharField(max_length=100)
    regular_hourly_rate = models.DecimalField(max_digits=5, decimal_places=2, default=5)
    overtime_hourly_rate = models.DecimalField(max_digits=5, decimal_places=2, default=1.25)
    hours_worked = models.PositiveIntegerField(default=40)
    overtime_threshold = models.PositiveIntegerField(default=40)
    date = models.DateField()

    def __str__(self) -> str:
        return self.name
    

    