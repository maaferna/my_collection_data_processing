from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.http import JsonResponse
import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
import ssl
import os


from .forms import *
from .models import *
from .utils import *


#External libraries
import requests
from bs4 import BeautifulSoup
import tweepy
import xml.etree.ElementTree as ET
from decouple import config

# Create your views here.

def index(request):
    context = {}
    return render(request, "index.html", context)

def calculate_overtime(request):
    if request.method == 'POST':
        form = OvertimeForm(request.POST)
        if form.is_valid():
            employee = form.save()
            return render(request, 'overtime_result.html', context={'employee':employee})
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


#Create a simple scraper to get information from a web page.
def scrape_data(request):
    # URL of the web page you want to scrape
    url = "https://portfolio-mparraf.herokuapp.com"
    module_dir = os.path.dirname(__file__)
    parent_directory = os.path.dirname(module_dir)

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    if response.status_code == 200:
        #Module to analyze files with text formatting HTML.
        soup = BeautifulSoup(response.content, 'html.parser')

        #Extract data of links referenced in this web page. This code pre-processes data and sends to the template an object that contains a list of anchors.
        links = [{ 'data': url +link.get('data'), 'title': link.get('title') } for link in soup.find_all('object')]
        print(links)
        # Render scraped data of all links in this web page.
        context = { 'links': links }
        return render(request, 'scraper/scraped_links.html', context)
    else:
        # Handle the case where the request was not successful
        return render(request, 'scraper/scraper_error.html')
    

def get_tweets(request):
    # Authenticate with Twitter API
    print(settings.TWITTER_BEARER_TOKEN)
    client = tweepy.Client(settings.TWITTER_BEARER_TOKEN, 
                           access_token=settings.TWITTER_API_KEY,
                            access_token_secret=settings.TWITTER_ACCESS_TOKEN,
                            consumer_key=settings.TWITTER_API_SECRET_KEY,
                            consumer_secret=settings.TWITTER_ACCESS_TOKEN_SECRET)
    tweet_text = 'This is a tweet from my Django app using Tweepy and Twitter API v2.'

    # Create the tweet
    tweet = client.create_tweet(text=tweet_text)
    tweet_id = tweet.id_str  # Get the ID of the created tweet
    response_data = {'success': True, 'tweet_id': tweet_id}
   
    return JsonResponse(response_data)

def xml_budget(request):
    url = 'http://www.dipres.gob.cl/597/articles-155375_doc_xml.xml'
    module_dir = os.path.dirname(__file__)
    parent_directory = os.path.dirname(module_dir)

    # Ignore SSL certificate errors
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # Desired value to match
    desired_value = "PRESIDENCIA DE LA REPÚBLICA"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            xml_data = response.content
            root = ET.fromstring(xml_data)
            print(root)
    except:
        xml_file_path = parent_directory + 'static/datasets/Presupuesto_2017.xml' 
        tree = ET.parse(xml_file_path)
        #root = tree.getroot()
        root = ET.fromstring(xml_data)  
    
    # Find all <cabecera> and <cuerpo> elements
    cabecera_elements = root.findall(".//cabecera")

    # Initialize a list to store "monto_pesos" values
    monto_pesos_values = [] 

    for cabecera in cabecera_elements:
        nombre_element = cabecera.find("nombre")
        # Check if "nombre" element exists and its text matches the desired value
        if nombre_element is not None and nombre_element.text.strip() == desired_value.strip():
            print(nombre_element.text)
            # Extract the text from the <nombre> element in <cabecera>
            cuerpo_elements = cabecera.findall("cuerpo")
            print(cuerpo_elements)
            # Iterate through "cuerpo" elements and extract "monto_pesos" values
            for cuerpo in cuerpo_elements:
                monto_pesos_element = cuerpo.find("monto_pesos")
                if monto_pesos_element is not None:
                    monto_pesos_values.append(monto_pesos_element.text)

    # Now, "monto_pesos_values" contains all "monto_pesos" values associated with the "cabecera" elements
    # where the "nombre" field matches the desired value
    for monto_pesos in monto_pesos_values:
        print(monto_pesos)
    
    print(monto_pesos_values)

    # Access the children elements within 'matriz'
    for child in root:
        # You can access child elements and their data here
        if child.tag == 'cabecera':
            for element in child:
                if element.tag == 'nombre' and element.text == 'SENADO':
                    name = element.text
                    print(name)

                # Add more conditions for other elements within 'cabecera'


    return render(request, 'budget_xml/index.html')

