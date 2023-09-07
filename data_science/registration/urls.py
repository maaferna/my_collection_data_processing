from django.urls import path
from .views import *

urlpatterns = [
    path('login', view_login, name='login'),
    path('logout', view_logout, name="logout"),
    path('register', view_register, name="register"),
    path('contact/', contact_view, name='contact_me'),

]