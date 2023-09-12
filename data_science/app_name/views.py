from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.http import JsonResponse
import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
from lxml import etree

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

def xml_books(request):
    #If It need acces altrough url
    url = ''
    module_dir = os.path.dirname(__file__)
    parent_directory = os.path.dirname(module_dir)

    # Ignore SSL certificate errors
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # Desired value to match
    desired_value = "ARTURO PEREZ-REVERTE"
    xml_file_path = parent_directory + '/static/datasets/books.xml'
    print(xml_file_path)
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except:
        print("Error to upload file")
    #response = requests.get(url)
    #if response.status_code == 200:
        #xml_data = response.content
        #root = ET.fromstring(xml_data)
        #root2 = etree.fromstring(xml_data)
    

    '''
    # Load and compile the XSD schema
    #xsd = etree.XMLSchema(etree.parse(xml_file_path))

    # Validate the XML document against the XSD schema
    is_valid = xsd.validate(root2)

    if is_valid:
        print("XML document is valid against the schema.")
        # Print the schema as a string
        print(etree.tostring(xsd.schema, pretty_print=True).decode('utf-8'))
    else:
        print("XML document is not valid against the schema.")
    '''

    books = [] 

    # Iterate through all book elements
    for book in root.findall(".//item"):
        author_element = book.find("auth")
        if request.GET.get("author_name", None) != None:
            author_name = request.GET.get("author_name")
            # Check if the author element exists and its text contains the filter substring
            if author_element is not None and author_name.strip().lower() in author_element.text.strip().lower():
                # Extract the data element and store in dictionary variable
                book_info = {
                    "id": book.find("isbn").text,
                    "title": book.find("book").text,
                    "language": (book.find("lang").text).capitalize(),
                    "price": book.find("euro").text,
                    "publish_date": book.find("year").text,
                    "description": book.find("about").text,
                    "publisher": (book.find("publ").text).capitalize(),
                    "tags": book.find("tags").text,
                    "img": 'https://' + book.find("img_url").text,
                    "page": book.find("page").text
                }
                # Store dictionarty to books list.
                books.append(book_info)
        
        context = {'books': books}


    return render(request, 'xml/index.html', context)

