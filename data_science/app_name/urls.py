from django.urls import path
from .views import *

urlpatterns = [
    path('', index, name='home'),
    path('overtime/', calculate_overtime, name='overtime'),
    path('overtime/result/', result_overtime, name='result_overtime'),
    path('overtime/list/', overtime_list, name='list_overtime'),
]