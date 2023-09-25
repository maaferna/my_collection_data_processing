import os
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

module_dir = os.path.dirname(__file__)
parent_directory = os.path.dirname(module_dir)

ContinentDict = {'China': 'Asia',
                     'United States': 'North America',
                     'Japan': 'Asia',
                     'United Kingdom': 'Europe',
                     'Russian Federation': 'Europe',
                     'Canada': 'North America',
                     'Germany': 'Europe',
                     'India': 'Asia',
                     'France': 'Europe',
                     'South Korea': 'Asia',
                     'Italy': 'Europe',
                     'Spain': 'Europe',
                     'Iran': 'Asia',
                     'Australia': 'Australia',
                     'Brazil': 'South America'}