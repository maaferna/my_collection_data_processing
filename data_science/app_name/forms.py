from django import forms
from django.forms.widgets import NumberInput
from .models import *
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit

class DateInput(forms.DateInput):
    input_type = 'date'

class OvertimeForm(forms.ModelForm):
    class Meta:
        model = Overtime
        fields = '__all__'
        widgets = {'date': DateInput()}

    widgets = {
        'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Input Name of Specialization', }),
        'regular_hourly_rate': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Input official descriptionn'}),
        'overtime_hourly_rate': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Input official descriptionn'}),
        'hours_worked': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Input official descriptionn'}),
        'overtime_threshold': forms.NumberInput(attrs={'class': 'form-control'}),
        'date': DateInput(),
    }