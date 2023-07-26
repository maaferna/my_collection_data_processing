from django.shortcuts import render, redirect

# Create your views here.
from .forms import *
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.models import User, Group
from django.contrib.auth.forms import AuthenticationForm
from .forms import *
from django.http import JsonResponse, HttpResponseRedirect
from django.urls import reverse

#Add permission to the views
from django.contrib.auth.decorators import login_required, permission_required

def view_login(request):
  if request.method == "POST":
    form = AuthenticationForm(request, data=request.POST)
    if form.is_valid():
      username = form.cleaned_data.get('username')
      password = form.cleaned_data.get('password')
      user = authenticate(username=username,
      password=password)
      if user is not None:
        login(request, user)
        messages.info(request, f"Iniciaste sesión como: {username}.")
        return HttpResponseRedirect('/')
      else:
        messages.error(request,"Invalido username o password.")
    else:
      messages.error(request,"Invalido username o password.")
  form = AuthenticationForm()
  return render(request=request, template_name="registration/login.html",context={"login_form":form})


def view_register(request):
  if request.method == "POST":
    form = RegistroUsuarioForm(request.POST)
    if form.is_valid():
      user = form.save()
      user.groups.add(Group.objects.get(name="visualizar_catalogo"))
      login(request, user)
      messages.success(request, "Registrado Satisfactoriamente." )
      return HttpResponseRedirect('/')
    messages.error(request, "Registro invalido. Algunos datos ingresados no son correctos")
  form = RegistroUsuarioForm()
  return render (request=request, template_name="registration/registro.html", context={"register_form":form})


def view_logout(request):
  logout(request)
  messages.info(request, "Se ha cerrado la sesión satisfactoriamente.")
  return HttpResponseRedirect('') 


def contact_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, 'success.html')
    form = ContactForm
    context = {'form': form}
    return render(request, 'contact.html', context)
