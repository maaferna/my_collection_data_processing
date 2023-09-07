from django.shortcuts import render
from .forms import *
# Create your views here.

def index(request):
    context = {}
    return render(request, "index.html", context)