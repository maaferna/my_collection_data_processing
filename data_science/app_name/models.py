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

    @property
    def calculate_overtime_pay(self):
        if self.hours_worked <= self.overtime_threshold:
            # No overtime, pay at regular rate
            total_pay = self.hours_worked * self.regular_hourly_rate
        else:
            # Calculate regular pay and overtime pay separately
            regular_pay = self.hours_worked * self.regular_hourly_rate
            overtime_hours = self.hours_worked - self.overtime_threshold
            overtime_pay = self.regular_hourly_rate * self.overtime_hourly_rate * overtime_hours
            total_pay = regular_pay + overtime_pay

        return total_pay
    