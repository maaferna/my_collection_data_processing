from django.shortcuts import render
from .forms import *
from .models import *
from .utils import *
# Create your views here.

def index(request):
    context = {}
    return render(request, "index.html", context)

def calculate_overtime(request):
    if request.method == 'POST':
        form = OvertimeForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, 'success.html')
    else:
        form = OvertimeForm()

    return render(request, 'overtime_form.html', {'form': form})

def overtime_list(request):
    employees = Overtime.objects.all()
    context = {'employees':employees, }
    return render(request, 'overtime_list.html', context)

def result_overtime(request, employee_id):
    
    try:
        employee = Overtime.objects.get(pk=employee_id)
        overtime_pay = calculate_overtime_pay(employee)
        context = { 'employee': employee, 'overtime_pay': overtime_pay}
        return render(request, 'overtime_result.html', context)
    except Overtime.DoesNotExist:
        return render('overtime_result.html')
    