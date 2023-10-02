from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.http import JsonResponse
import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
from lxml import etree
from collections import Counter
import re
# To generate images
from io import BytesIO
import base64
import datetime

import ssl
import os
import string

from .forms import *
from .models import *
from .utils import *


#External libraries
import requests
from bs4 import BeautifulSoup
import tweepy
#from decouple import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

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
        title_element = book.find("book")
        if request.GET.get("author_name", None) != None and request.GET.get("title", None) != None:
            author_name = request.GET.get("author_name")
            title = request.GET.get("title")
            # Check if the author element exists and its text contains the filter substring
            if author_element is not None and author_name.strip().lower() in author_element.text.strip().lower() and title.strip().lower() in title_element.text.strip().lower():
                # Extract the data element and store in dictionary variable
                try:
                    book_info = {
                        "id": book.find("isbn").text,
                        "title": book.find("book").text,
                        "language": book.find("lang").text.capitalize(),
                        "price": book.find("euro").text,
                        "publish_date": book.find("year").text,
                        "description": book.find("about").text,
                        "publisher": book.find("publ").text.capitalize(),
                        "tags": book.find("tags").text,
                        "img": 'https://' + book.find("img_url").text,
                        "page": book.find("page").text
                    }
                except:
                    continue
                # Store dictionarty to books list.Pass if book register not contain text
                if book_info["description"] == None:
                    pass
                else:
                    books.append(book_info)
                
        
        context = {'books': books, 'author': author_name}


    return render(request, 'xml/index.html', context)

def regex(request):
    module_dir = os.path.dirname(__file__)
    parent_directory = os.path.dirname(module_dir)
    file_path = parent_directory + '/static/datasets/Regex sample file.txt'
    # Open and read the text file
    with open(file_path, 'r') as file:
        contents = file.read()

    with open(file_path, 'r') as file:  
        contents_by_line = file.readlines()
        
    # Use regular expression to find all numbers
    numbers = re.findall(r'\d+', contents)
    # Use regular expression to find all line to contain only a number.
    line_with_digits = []
    for line in contents_by_line:
        # Use regular expression to check if the line is a number
        if re.match(r'^\d+$', line.strip()):
            line_with_digits.append(line.strip())

    #Define Regex expresion
    url_pattern = r'\b(?:https?://|www\.)\S+\b'

    #Search in text to find all the URL that begin with www. / https or http.
    found_urls = []
    found_urls_with_startwish = []
    try:
        matches = re.finditer(url_pattern, contents)
        for match in matches:
            found_urls.append(match.group(0))
        
        #Find line that begin with some text
        for line in contents_by_line:
            line.rstrip()
            if line.startswith("Terminology:"):
                found_urls_with_startwish.append(line)

    except FileNotFoundError:
        print("File not found.")

    print(found_urls_with_startwish)
    context = {}

    return render(request, 'regex/index.html', context)


def file_analytic(request):
    file_path = parent_directory + '/static/datasets/Regex sample file.txt'
    # Open and read the text file
    languages = ['Java', 'C++', 'PHP', 'Ruby', 'Basic', 'Perl', 'JavaScript', 'Python']
    # Initialize the dictionary with zero counts for each language
    analytics = {lan: 0 for lan in languages}

    with open(file_path, 'r') as file:
        contents = file.readlines()

    for line in contents:
        for lan in languages:
            if lan in line:
                analytics[lan] += 1

    # Sort the dictionary by value
    lst = list()
    for key, val in list(analytics.items()):
        lst.append((val, key))

    lst.sort(reverse=True)

    for key, val in lst[:10]:
        print(key, val)
    
    context = {}

    return render(request, 'regex/index.html', context)


def data_cleaning(request):
    file_path = parent_directory + '/static/datasets/olympics.csv'
    df = pd.read_csv(file_path, index_col=0, skiprows=1)
    df_sorted = df.sort_values(by='Combined total', ascending=False)  # Change 'ascending' to False retrieve the countries with more medals gotten.
    # The type of medals area is defined with a number (01: Gold, 02: Silver, 03:Bronze). The next line was used to rename the column name and retrieve data with a better description.
    for col in df_sorted.columns:
        if col[:2]=='01':
            df_sorted.rename(columns={col:'Gold'+col[4:]}, inplace=True)
        if col[:2]=='02':
            df_sorted.rename(columns={col:'Silver'+col[4:]}, inplace=True)
        if col[:2]=='03':
            df_sorted.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
        if col[:1]=='№':
            df_sorted.rename(columns={col:'#'+col[1:]}, inplace=True)
    
    # To extract the ID of country was applied the split() function in the column name.
    names_ids = df_sorted.index.str.split('\s\(') # split the index by '('
    df_sorted.index = names_ids.str[0] # the [0] element is the country name (new index) 
    df_sorted['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)
    df = pd.DataFrame(df_sorted)
    # Rename the column at position 0 (Column1) to 'New Name'
    # Assuming you have your DataFrame df already defined
    df_html = df[1:].to_html(classes='table table-bordered table-striped', index=True)
    answer= []
    max_difference_gold = abs(df['Gold'][1:] - df['Gold.1'][1:]).idxmax()
    # Filter the DataFrame to get the row for "Max Gol difference"
    gb_row = df.loc[max_difference_gold]
    # Get the 'Gold_Difference' value for "Max Gol difference"
    gold_difference_gb = gb_row['Gold'] - gb_row['Gold.1']
    # Add a condition to filter only Countries that gained at least one Gold medal for each season (Summer Winter).
    condition = (df['Gold'] >=1) & (df['Gold.1'] >=1)
    new_df = df[condition]

    # This function generates a Series named 'Points.' This Series represents a weighted value system, where each gold medal (Gold.2) is assigned a weight of 3 points, silver medals (Silver.2) carry 2 points, and bronze medals (Bronze.2) contribute 1 point. The function should return the resulting column as a Series object, with the country names serving as the indices.
    df['Points'] = pd.Series(df['Gold.2'] * 3 + df['Silver.2'] * 2 + df['Bronze.2'])

    answer.append(max_difference_gold)
    answer.append(gold_difference_gb)
    context = {'data': df_html}

    # The country that has won the most gold medals in summer games is ____. Using idxmax()
    print(df['Gold'][1:].idxmax())
    print((abs(df['Gold'][1:] - df['Gold.1'][1:])).idxmax())
    print((abs(new_df['Gold'] - new_df['Gold.1'])/new_df['Gold.2']).idxmax())
    print(df['Points'][1:])
    #
    return render(request, 'pandas/data-cleaning.html', context)

def find_min_max(row):
    # Select the columns of interest
    columns_of_interest = ['POPESTIMATE2010', 'POPESTIMATE2011', 'POPESTIMATE2012', 'POPESTIMATE2013', 'POPESTIMATE2014', 'POPESTIMATE2015']
    # Find the minimum and maximum values for the selected columns in the row
    min_value = row[columns_of_interest].min()
    max_value = row[columns_of_interest].max()

    return pd.Series({'MIN_POP': min_value, 'MAX_POP': max_value, 'DIF_POP': max_value-min_value})


def data_cleaning_census(request):
    file_path = parent_directory + '/static/datasets/census.csv'
    df = pd.read_csv(file_path)
    # Identify the state that encompasses the greatest number of counties. The {{ result.0 }} is the State with more Counties in the United State they are {{ result.1 }}
    greatest_counties = df.groupby(['STNAME']).sum()['COUNTY'].idxmax()
    quantity_greatest_county = df.groupby(['STNAME']).sum()['COUNTY'].max()
    print(greatest_counties, quantity_greatest_county)
    # Find the top ten populous states in descendent order, using the column CENSUS2010POP. To make more complex was filtered with code 50 to group by Counties
    # The key for SUMLEV is as follows:
    # 040 = State and/or Statistical Equivalent
    # 050 = County and /or Statistical Equivalent
    condition_state = df['SUMLEV'] == 50  # Select only STATE, Drop rows that contain state data
    new_condition_state = df[condition_state]
    # Sort the county-level data by state ('STNAME') and population ('CENSUS2010POP') in descending order
    most_population_state = new_condition_state.sort_values(['STNAME', 'CENSUS2010POP'], ascending=[True, False])
    # Calculate the total population for each state and sort them in descending order
    population = most_population_state.groupby('STNAME').agg('sum').sort_values('CENSUS2010POP', ascending=False)
    highest_state = population.head(10).reset_index()

    # The next task is to identify the City Name with the largest absolute change in population between the years 2010 and 2015. 
    # The code calculates the population change for each county and selects the top ten city with the highest change.  

    df_pop = df[condition_state]
    # Apply the function to each row to find the minimum and maximum values
    min_max_values = df_pop.apply(find_min_max, axis=1)

    # Concatenate the resulting DataFrame with the original DataFrame
    df = pd.concat([df_pop, min_max_values], axis=1)

    pop_diff = df[['CTYNAME', 'DIF_POP']].sort_values('DIF_POP' ,ascending=False)

    #Query the dataset to retrieve parameters passed through GET methods in the URL for columns related to regions and the starting city name.
    # Validate if there has been an increase in population from 2014 to 2015 for the specified counties using conditional filtering and string operations. The code then extracts and returns these counties as a DataFrame."
    
    #filter region condition
    condition_region = (df['SUMLEV']==50) & (df['REGION'] ==1) | (df['REGION'] ==2)
    region_df = df.copy()
    region_df = region_df[condition_region]
    #Filter county name that start with some text.
    condition_starts_name = region_df['CTYNAME'].str.startswith('Washington')
    start_city = region_df[condition_starts_name]
    #Validate increase Population 2015/2016
    condition_comparation_population =  start_city['POPESTIMATE2015'] > start_city['POPESTIMATE2014']
    fiend_counties =  start_city[condition_comparation_population]
    print(fiend_counties[['STNAME', 'CTYNAME']])


    context = {}
    return render(request, 'pandas/data-cleaning.html', context)

def data_read_excel_file(request):
    file_path = parent_directory + '/static/datasets/Energy Indicators.xls'
    file_path2 = parent_directory + '/static/datasets/world_bank.csv'
    file_path3 = parent_directory + '/static/datasets/scimagojr-3.xlsx'
    
    #This project involves loading energy-related data from the files with xls, csv, xlsx extension
    
    # Read energy data from an Excel file, skipping 17 rows from the top
    # and 38 rows from the bottom to exclude header and footer information.
    energy = pd.read_excel(file_path, skiprows=17, skipfooter=38)

    # Drop the first and second columns of the 'energy' DataFrame to remove unnecessary data.
    energy.drop(energy.columns[[0, 1]], axis=1, inplace=True)

    # Rename the columns of the 'energy' DataFrame for clarity and consistency.
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    
    # Replace '...' entries in the 'Energy Supply' column with NaN (Not a Number) values to represent missing or incomplete data.
    # Should be import numpy as np in the header of file
    energy.loc[energy['Energy Supply'] == '...'] = np.NaN
    # Drop rows with missing data (NaN values) from the 'energy' DataFrame.
    energy = energy.dropna()

    # Convert Energy to gigajoules
    energy['Energy Supply'] *= 1000000

    # Rename entries for Country Columns
    energy['Country'] = energy['Country'].replace('Iran (Islamic Republic of)', 'Iran')
    energy['Country'] = energy['Country'].str.replace('\(.*\)','')
    energy['Country'] = energy['Country'].str.replace('\,\ ', ' ')
    energy['Country'] = energy['Country'].str.replace(r'\d+', '', regex=True)
    energy['Country'] = energy['Country'].replace('Republic of Korea', 'South Korea')
    #energy['Country'] = energy['Country'].replace('China2', 'China')
    #energy['Country'] = energy['Country'].replace('France6', 'France')
    energy['Country'] = energy['Country'].replace('United States of America', 'United States')
    energy['Country'] = energy['Country'].replace('United Kingdom of Great Britain and Northern Ireland','United Kingdom')
    energy['Country'] = energy['Country'].replace('China, Hong Kong Special Administrative Region3', 'Hong Kong')

    # Read Gross domestic product, that represent is a measure of the size and health of a country's economy over a period of time   
    # Install in virtual enviroment pipenv install xlrd
    gdp = pd.read_csv(file_path2, skiprows=4)
    # Rename Country Name for consitency with energy DataFrame
    gdp['Country Name'] = gdp['Country Name'].replace({'Korea, Rep.':'South Korea','Iran, Islamic Rep.':'Iran','Hong Kong SAR, China':'Hong Kong'})
    # Rename the 'Country Name' column to 'Country' in the 'GDP' DataFrame for consistency with Country column in energy DataFrame.
    gdp = gdp.rename(columns={'Country Name':'Country'})

    # Install in virtual enviroment pipenv install openpyxl
    power_energy = pd.read_excel(file_path3)
    # Select top 10 countries of Ranking
    power_energy_ranking = power_energy[0:15]

    # Merge the 'gdp' and 'power_energy_ranking' DataFrames using an inner join on the 'Country' column.
    gdp_merge_power_energy_ranking = pd.merge(gdp, power_energy_ranking,how='inner',left_on='Country', right_on='Country')

    # Merge the 'gdp_merge_power_energy_ranking' and 'energy' DataFrames using an inner join on the 'Country' column.
    merges_data = pd.merge(gdp_merge_power_energy_ranking, energy,how='inner',left_on='Country', right_on='Country')

    # Re-Define index how Contry columns 
    merges_data = merges_data.set_index('Country')

    data = merges_data[['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
    
    data = data.sort_values('Rank')
    
    answer = []
    # Retrieve the Avarage energy between 2006 and 2015
    top15_avg_condition = data[['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
    top15_avg_rank = top15_avg_condition.mean(axis=1).sort_values(ascending=False)
    
    # In this analysis, we aim to calculate the extent of GDP fluctuation over a 10-year duration for the nation holding the 4th position in terms of its average GDP.
    data['avg_gdp'] = data[['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']].mean(axis=1)
    data.sort_values("avg_gdp", ascending=False, inplace=True)
    abs_value = abs(data.iloc[3]['2015']- data.iloc[3]['2006'])
    country = data.index[3]
    answer.append(abs_value)
    answer.append(country)

    # Retrieve the country with the lowest percentage of renewable energy and reporting the name of the country along with its corresponding percentage.
    lower_index = data['% Renewable'].idxmin()
    lower_percentage = round(data['% Renewable'].min(), 1)
    lower_percentage = f"{lower_percentage}%"

    answer.append(lower_index)
    answer.append(lower_percentage)

    # Calculate a new column representing the ratio of Self-Citations to Total Citations. Determine the maximum value within this new column and identify the country with the highest ratio.
    data['Ratio'] = data['Self-citations'] / data['Citations']
    lower_index_ratio = data['Ratio'].idxmin()
    lower_percentage_ratio = round(data['Ratio'].min(), 3)

    # Generate a new column to estimate the population by using the Energy Supply and Energy Supply per capita values. Determine the third most populous country based on this population estimate.
    # Convert columns to numeric and handle non-numeric values
    data['Energy Supply'] = pd.to_numeric(data['Energy Supply'], errors='coerce')
    data['Energy Supply per Capita'] = pd.to_numeric(data['Energy Supply per Capita'], errors='coerce')

    # Fill missing values with 0 or other appropriate values
    data['Energy Supply'].fillna(0, inplace=True)
    data['Energy Supply per Capita'].fillna(0, inplace=True)

    # Perform the division and round
    data['PopEstimate'] = (data['Energy Supply'] / data['Energy Supply per Capita']).round()

    data.sort_values('PopEstimate', ascending=False, inplace=True)
    estimate_pop = data.iloc[2].name  

    # Generation a new column that estimates the ratio of citable documents to the population (per person). Calculate the correlation between this ratio and the energy supply per capita using Pearson's correlation coefficient via the .corr() method.
    data['PopEstimate'] = data['Energy Supply'] / data['Energy Supply per Capita']
    data['PopCitableDocuments'] = data['Citable documents'] / data['PopEstimate']
    correlation = data['PopCitableDocuments'].astype(float).corr(data['Energy Supply per Capita'].astype(float))
    # A correlation coefficient of 0.79 suggests a strong positive linear relationship between the ratio of citable documents to the population (per person) and the energy supply per capita. In other words, as the energy supply per capita increases, there tends to be a corresponding increase in the ratio of citable documents to the population. This indicates that countries with higher energy supply per capita tend to have more citable documents per person, implying a potential connection between energy availability and scientific research output.
    print(correlation)

    # Use the matplotlib built-in function plot() to visualize the relationship between Energy Supply per Capita and Citable Documents per Capita. Install the matplotlib, BytesIO, & base64  library in your virtual environment and include the necessary import statement in the file's header.
    data.plot(x='PopCitableDocuments', y='Energy Supply per Capita', kind='scatter') 
    
    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()    
    # Encode the image to base64
    graphic = base64.b64encode(image_png).decode()
    
    # Using DataFrame tools, set up a column containing a '1' if a country's % Renewable value matches or exceeds the median value among all countries within the top 15 rankings. Conversely, assign a '0' if the country's % Renewable value falls below the median. The resulting output will be a series named "High Renewable," and its index will be sorted in ascending order based on the country's rank.
    mediana_value = data['% Renewable'].median()
    data['HighRenewable'] = data.apply(lambda x: 1 if x['% Renewable'] >= mediana_value else 0, axis=1)
    data.sort_values('Rank',ascending=True,inplace=True)
    answer.append(mediana_value)

    # Utilize the provided dictionary, defined as "ContinentDict," to group countries by their respective continents. Subsequently, construct a DataFrame that presents key statistics for each continent, including the sample size (number of countries), the sum, mean, and standard deviation of the estimated population of the countries within that continent.
    
    data_by_continent = pd.DataFrame(columns=['size', 'sum', 'mean', 'std'])
    for idx, name in data.groupby(ContinentDict):
        data_by_continent.loc[idx] = [len(name), name['PopEstimate'].sum(), name['PopEstimate'].mean(), name['PopEstimate'].std()]

    # Divide the '% Renewable' values into five distinct bins. Afterward, categorize the 'Top15' dataset by both continent and the newly defined bins for '% Renewable.' Determine the count of countries in each of these combined groupings.

    # The desired outcome is a Series that incorporates a MultiIndex structure, first based on the continent, and then further sub-divided by the bins representing '% Renewable.' Exclude groups that have no countries within them.
    data['ByContinent'] = data.index.to_series().map(ContinentDict)
    data['SubGroups'] = pd.cut(data['% Renewable'], 5)
    data_by_continent_bins = data.groupby(['ByContinent', 'SubGroups']).size()
    data_by_continent_bins = data_by_continent_bins[data_by_continent_bins > 0]
   
    # Create the bubble chart
    plt.figure(figsize=(12, 8))  # Set the figure size

    # Scatter plot with bubbles
    plt.scatter(data['Rank'], data['Energy Supply'], s=data['Energy Supply per Capita'], c='b', alpha=0.5)

    # Labeling axes and title
    plt.xlabel('Rank')
    plt.ylabel('Energy Supply')
    plt.title('Bubble Chart: Rank vs. Energy Supply vs. Energy Supply per Capita')

    # Adding a color bar for the size legend
    plt.colorbar(label='Energy Supply per Capita')
    plt.grid(True)
   
   # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()    
    # Encode the image to base64
    graphic2 = base64.b64encode(image_png).decode()  
    print(data)



    context = {'graphic': graphic , 'graphic2': graphic2}
    return render(request, 'pandas/data-cleaning.html', context)

def hypothesis_testing(request):
    file_path = parent_directory + '/static/datasets/university_towns.txt'
    file_path2 = parent_directory + '/static/datasets/City_Zhvi_AllHomes.csv'
    file_path3 = parent_directory + '/static/datasets/gdplev.xls'
    answer = []
    # Obtain data from datasets usin Pandas 
    df_towns = pd.read_fwf(file_path, header=None).rename(columns={0: 'State'})
    df_gdp = pd.read_excel(file_path3, skiprows=7).rename(columns={'Unnamed: 4':'Year_Quartile','Unnamed: 6':'GDP'})
    df_housing = pd.read_csv(file_path2)
    df_gdp_end = df_gdp.copy()
    df_gdp_bottom = df_gdp.copy()
    df_gdp = df_gdp[211:]
    df_towns_hypothesis = df_towns.copy()

    df_university_towns = pd.DataFrame(columns=['State', 'RegionName'])
    pattern = r'\[edit\]'
    pattern_edit = '[edit]'
    df_row = []
    for line in df_towns['State']:
        if pattern_edit in line:
            town = None
            ed = re.search(pattern, line)
            state = line[:ed.start()].strip()
        else:
            if line == 'The Colleges of Worcester Consortium:':
                town = 'The Colleges of Worcester Consortium:'
            elif line == 'The Five College Region of Western Massachusetts:':
                town = 'The Five College Region of Western Massachusetts:'
            elif line == 'Faribault, South Central College':
                town = 'Faribault, South Central College'
            elif line == 'North Mankato, South Central College':
                town = 'North Mankato, South Central College'
            else:
                nam_end = re.search(r'[\(:]', line)
                if nam_end:
                    town = line[:nam_end.start()].strip()

        if town is not None and state is not None:
            df_row.append({'State': state, 'RegionName': town})
    
    df_university_towns = pd.concat([df_university_towns, pd.DataFrame(df_row)], ignore_index=True)

    # Provides the year and quarter of when the recession started. It is defined as two consecutive quarters of Gross Domestic Product (GDP) decline, concluding with two-quarters GDP growth.
    year_quartile_start = []
    for line in range(4, len(df_gdp)):
        if (df_gdp.iloc[line-4,6] > df_gdp.iloc[line-3,6]) and (df_gdp.iloc[line - 3, 6] > df_gdp.iloc[line - 2,6]):
            year_quartile_start.append(df_gdp.iloc[line-3,4])

    answer.append(year_quartile_start[0])

    # Provides the year and quarter of when the recession end time.
    start_index = df_gdp_end[df_gdp_end['Year_Quartile'] == year_quartile_start[0]].index.to_list()
    df_gdp_end = df_gdp_end[start_index[0]:]
    year_quartile_end = []
    for line in range(2, len(df_gdp_end)):
        if (df_gdp_end.iloc[line - 4, 6] < df_gdp_end.iloc[line - 3, 6]) and (df_gdp_end.iloc[line - 3, 6] < df_gdp_end.iloc[line - 2, 6]):
            year_quartile_end.append(df_gdp_end.iloc[line - 2, 4])

    answer.append(year_quartile_end[0])
    end_index = df_gdp_end[df_gdp_end['Year_Quartile'] == year_quartile_end[0]].index.to_list()
    
    # The recession bottom represents the quarter within a recession period that records the lowest GDP.
    df_gdp_bottom = df_gdp_bottom[start_index[0]:end_index[0]]
    df_gdp_bottom.reset_index(drop=True, inplace=True)
    bottom_idx = df_gdp_bottom['GDP'].idxmin()
    bottom = df_gdp_bottom.iloc[bottom_idx,4]

    # This process involves converting housing data into quarterly intervals and presenting it as a DataFrame containing mean values. The resulting DataFrame will possess a multi-index structure, combining the "State" and "RegionName" as index levels.
    # The resulting DataFrame will have columns ranging from 2000q1 to 2016q3, providing a comprehensive overview of housing data trends over this period.
    df_housing['State'] = df_housing['State'].map(states)
    df_housing.drop(['RegionID','CountyName','Metro','SizeRank'], axis=1, inplace=True)
    df_housing = df_housing.set_index(['State','RegionName'])
    col_init = df_housing.columns.get_loc('2000-01')
    df_housing = df_housing.iloc[:,col_init:]
    quartile_id = ['q1','q2','q3','q4']
    df_quarters_name = []
    for y in range(2000, 2017):
            for q in quartile_id:
                df_quarters_name.append(str(y)+q)
    df_quarters_name.pop()

    periodo = df_housing.columns
    df_quarter_mean = []
    for p in range(0,len(periodo),3):
        df_sub = df_housing.iloc[:, p:p+3]
        quarter_mean = df_sub.mean(axis = 1).round(2)
        df_quarter_mean.append(quarter_mean)

    df_mean = df_quarter_mean[0]
    for i in range(1,len(df_quarter_mean)):
        df_mean = pd.concat([df_mean,df_quarter_mean[i]], axis=1)

    df_mean.columns = df_quarters_name
    # We start by creating a new dataset that shows the changes in housing prices between the beginning of the recession and the recession's lowest point.
    # Perform a t-test to compare housing price trends in university towns with those in non-university towns. The t-test helps us determine whether there is a significant difference between these two groups.
    # Evaluate the alternative hypothesis, which posits that the two groups have different housing price trends. Specifically, we assess whether this hypothesis is true or not.
    # Calculate the p-value using the scipy.stats.ttest_ind() function. The p-value is a measure of the confidence level in our results.

    df_mean['recession_diff'] = df_mean[year_quartile_start[0]] - df_mean[bottom]
    df_mean['with_university'] = False
    df_housing_towns_with = pd.merge(df_mean,df_university_towns, how='inner', on = ['State','RegionName'])
    df_housing_towns_with['with_university'] = True

    df_ttest_with = df_housing_towns_with['recession_diff'].dropna()
    df_mean.drop('with_university',axis='columns', inplace=True)
    df_housing_towns_with.drop('with_university',axis='columns', inplace=True)

    df_housing_towns_non = pd.concat([df_mean, df_housing_towns_with]).drop_duplicates(keep=False)

    df_ttest_non = df_housing_towns_non['recession_diff'].dropna()

    st, p = ttest_ind(df_ttest_with, df_ttest_non, nan_policy='omit')

    if st < 0.01:
        st = True

    if df_ttest_with.mean() < df_ttest_non.mean():
        best = "university towns"
    else:
        best = "non-university town"
    
    print(st, p, best)
    context = {}
    return render(request, 'pandas/data-cleaning.html', context)

def daily_climate(request):
    # Climate Data Daily IDN Chili daily climate data from Agrometeorología (Red agromemtrológica INIA) 2010 to 2023. Cover: Pixabay
    file_path = parent_directory + '/static/datasets/agrometeorologia-chillan2013-2023.csv'
    df = pd.read_csv(file_path, skiprows=5, header=0)

    # Define a list of columns that should only contain dates
    date_columns = ['Tiempo UTC-4']
    integer_columns = ['Temperatura del Aire Mínima ºC', 'Temperatura del Aire Máxima ºC']
    # Filter out rows with NaN values in date columns and reset the index
    df_cleaned = df.dropna(subset=date_columns, how='all').reset_index(drop=True)
    df_cleaned = df_cleaned.dropna(subset=integer_columns, how='all')
    df_cleaned.rename(columns={integer_columns[0]:'TMIN', integer_columns[1]:'TMAX'}, inplace=True)

    print(df_cleaned)
    # Begin by thoroughly studying the dataset documentation to gain a comprehensive understanding of its structure and contents.
    # Temperature Trends Visualization: Develop Python code that generates a line graph depicting the record high and record low temperatures for each day of the year, covering the period from 2005 to 2014. To enhance clarity, the area between the record high and record low temperatures for each day should be shaded.
    # Through this project, we aim to not only uncover temperature trends but also demonstrate the power of data visualization in making complex information accessible and engaging. Ultimately, the project will provide a valuable resource for understanding historical climate patterns in Chillán city.
    plt.figure(figsize=(16,10))
    plt.title('Record high and record low temperatures by day (period 2010-2023)',alpha=0.8)

    plt.plot(df_cleaned['TMAX'], c = 'red', label ='Record High')
    plt.plot(df_cleaned['TMIN'], c = 'blue', label ='Record Low')
    plt.gca().fill_between(range(len(df_cleaned)),df_cleaned['TMAX'],df_cleaned['TMIN'],facecolor='black',alpha=0.25)
    plt.legend(['Record High T°', 'Record Low T°'])

    plt.xlabel('Days')
    plt.ylabel('Temperature, (tenths C°)')

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.show()

 
    # Convert the 'Tiempo UTC-4' column to datetime format
    df_cleaned['Tiempo UTC-4'] = pd.to_datetime(df_cleaned['Tiempo UTC-4'], format='%d-%m-%Y')
    # Filter the DataFrame for the year 2022
    df_2022 = df_cleaned[df_cleaned['Tiempo UTC-4'].dt.year == 2022]
    # Find the maximum value of 'TMAX' for the year 2022
    max_tmax_2022 = df_2022['TMAX'].max()
    min_tmax_2022 = df_2022['TMIN'].min()
    # Highlighting Anomalies: Overlay a scatter plot with data from the year 2022. This scatter plot will emphasize any data points (representing temperature highs and lows) where the records set between 2005 and 2014 were surpassed in 2022. These points will provide valuable insights into notable climate anomalies.

    plt.figure(figsize=(16,10))
    plt.title('Record high and record low temperatures by day (year 2022)',alpha=0.8)

    plt.scatter(df_cleaned['Tiempo UTC-4'],df_cleaned['TMAX'], c = 'red', label ='Record High')
    plt.scatter(df_cleaned['Tiempo UTC-4'],df_cleaned['TMIN'], c = 'blue', label ='Record Low')

    plt.legend(['Record High T (2015)°', 'Record Low T° (2015)','Record High (period 2005-2014)','Record Low (period 2005-2014)'])

    plt.xlabel('Date')
    plt.ylabel('Temperature, (tenths C°)')

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    x = plt.gca().xaxis
    for item in x.get_ticklabels():
        item.set_rotation(45)

    # Convert the date strings to datetime objects for xlim
    xlim_start = pd.to_datetime('01-01-2010', format='%d-%m-%Y')
    xlim_end = pd.to_datetime('01-10-2023', format='%d-%m-%Y')

    plt.xlim([xlim_start, xlim_end])
        
    plt.axhline(y = max_tmax_2022, color='r', linestyle='-',label ='Record High (period 2005-2014)')
    plt.axhline(y = min_tmax_2022, color='b', linestyle='-',label ='Record Low (period 2005-2014)')

    plt.show()

    context = {}
    return render(request, 'pandas/data-cleaning.html', context)