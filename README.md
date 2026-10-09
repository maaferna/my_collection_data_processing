# Data Processing Collection — applied Python exercises, served live

Collection of data-science exercises built while completing data specializations
(pandas, NumPy, Matplotlib, scikit-learn, Beautiful Soup, regular expressions),
packaged as a Django project so every exercise is a **self-documenting live
page**: each one renders its own source code next to its computed results.

**▶ See it running — the collection is embedded in my portfolio:**
[portfolio-mparraf.herokuapp.com/data-science](https://portfolio-mparraf.herokuapp.com/data-science/)

## What's inside

| Area | Exercises |
|---|---|
| Regex | Text processing and analysis with regular expressions |
| Scraping | Website scraping with Beautiful Soup inside a Django view |
| Pandas | CSV processing, census data analysis, multi-source energy/GDP integration, XML book catalog |
| Visualization | Interactive climate data exploration, decision-support charts (Matplotlib rendered in-memory, injected as base64) |
| Machine learning | k-NN classification with scikit-learn |

No files are written to disk for the charts: figures are generated in memory
per request and embedded directly into the page.

## Stack

Python · Django · pandas · NumPy · Matplotlib · scikit-learn ·
Beautiful Soup · Pipenv

## Running locally

```bash
cd data_science
pipenv install            # or: pip install -r requirements.txt
cp .env.example .env      # fill in your own values — never commit .env
python manage.py migrate
python manage.py runserver
```

## Author

Marco Antonio Parra F. — [portfolio](https://portfolio-mparraf.herokuapp.com) ·
[LinkedIn](https://www.linkedin.com/in/marco-antonio-parra-82999337/)
