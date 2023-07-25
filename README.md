<h1>Django Project Base</h1>

<h2>Description</h2>
This project is used as a how base (template) to create new projects, considers the creation of a virtual environment (with Pipenv), installation of Django Framework, and initial migrations to create Auth's models.

<h6>Step by Step</h6>
Go to directory.
-Create virtual enviroment with Pipenv
<code>
    pipenv shell<
</code>
-Install libraries Django / django-dotenv
<code>
    pipenv install django django-dotenv
</code>
-Create project
<code>
    django-admin startproject project_name
</code>
-Apply first migrations:admin, auth, contenttypes, sessions
<code>
    python manage.py migrate
</code>

-In root directory project create files .env and gitignore
<code>
 touch .env .gitignore
</code>

-In the settings file change the SECRET_KEY locations, to store the key in the .env file. Include next code

<code>
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
SECRET_KEY = os.environ['SECRET_KEY']
</code>

<br>
<img height="50px" src="https://portfolio-mparraf.herokuapp.com/static/img/django.png" />
<br>


To store environment variables were created .env and .gitignore files, and install the library "python-dotenv" to manage the importation of these.





<h2>Installation</h2>

The project should be clone in local directory with next code line:

Pipenv should be installed with Python 3.8.
