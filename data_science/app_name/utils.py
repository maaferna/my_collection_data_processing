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

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 
          'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 
          'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 
          'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho',
        'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii',
        'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 
        'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico',
        'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 
        'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 
        'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 
        'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska',
        'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 
        'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island',
        'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 
        'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}