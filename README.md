<h1>Django Project Base</h1>

<h2>Description</h2>
This project is used as a how base (template) to create new projects, considers the creation of a virtual environment (with Pipenv), installation of Django Framework, and initial migrations to create Auth's models.
<br>
<img height="50px" src="https://portfolio-mparraf.herokuapp.com/static/img/django.png" />
<br>

<h6>Step by Step</h6>
Go to directory.
-Create virtual enviroment with Pipenv
<code>pipenv shell</code>
<br>
-Install libraries Django / django-dotenv
<code>pipenv install django django-dotenv</code>
<br>
-Create a requirements file to store the basic package for this project, use the next code.
<code>pip freeze > requirements.txt</code>

-Create project
<code>django-admin startproject project_name</code>
<br>

-Apply first migrations:admin, auth, contenttypes, sessions
<code>python manage.py migrate</code>
<br>
-In root directory project create files .env and gitignore
<code>touch .env .gitignore</code>

<br>
- To store environment variables were created .env and .gitignore files, and install the library "python-dotenv" to manage the importation of these.
-In the settings file change the SECRET_KEY locations, to store the key in the .env file. Include next code in settings:

<div style="background-color: blue"><p>python-dotenv</p></div>
<code>from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
SECRET_KEY = os.environ['SECRET_KEY']</code>

-In the .env file store key:
<code>SECRET_KEY="Insert here django key for this project"</code>

-In the .gitignorefile, include .env file to avoid deploy in Github Repository
<code>.env</code>

<h6>Schema</h6>
-This project considers how schema the creation of templates and static directories, in the root of the project. In the command line, go to the root directory and execute the next code:
<code>mkdir templates</code>
<code>mkdir static</code>
<code>mkdir static</code>
<code>cd static</code>
<code>mkdir css</code>
<code>mkdir js</code>

<h2>Installation</h2>

-The project should be clone in local directory in command line with next command:

<code>git clone git@github.com:maaferna/project_base_django.git</code> For SSH Protocol, Or
<code>git clone https://github.com/maaferna/project_base_django.git</code> HTTPS

-Create a virtual enviroment with pipenv package.
Pipenv should be installed with Python = "3.11" & Django==4.2.3, but this was used the requeriments file.
<code>pipenv shell</code>
Next, install the basic package for this project, use next code:
<code>pip install -r requirements.txt</code>
-Migrate database and create superuser
<code>python manage.py migrate
python manage.py createsuperuser</code>
-Personalize project and app names respectively. In urls.py file inside of project directory 

<code>urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('app_name')) #updated this name with real app
]</code>

The settings.py should be:

<code>INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'fontawesome_5',
    "bootstrap5",
    'crispy_bootstrap5',
    'crispy_forms',
    'rest_framework',
    'app_name.apps.AppNameConfig', #update this line with real name of apps, see in directory of initial app the apps.py file.
]</code>




