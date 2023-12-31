from django.urls import path
from .views import *

urlpatterns = [
    path('', index, name='home'),
    path('overtime/', calculate_overtime, name='overtime'),
    path('overtime/result/', result_overtime, name='result_overtime'),
    path('overtime/list/', overtime_list, name='list_overtime'),
    path('scrape/beutifulsoup/', scrape_data, name='scrape_data'),
    path('get-tweets/', get_tweets, name='get_tweets'),
    path('books-by-author/', xml_books, name='xml_books'),
    path('regex/', regex, name='regex'),
    path('file-analytics/', file_analytic, name='file_analytic'),
    path('pandas/data-cleaning/', data_cleaning, name='pandas-data-cleaning'),
    path('pandas/data-cleaning/census/', data_cleaning_census, name='pandas-census'),
    path('pandas/data-cleaning/excel_read/', data_read_excel_file, name='excel_read'),
    path('pandas/hypothesis/', hypothesis_testing, name='hypothesis'),
    path('data-visualization/', daily_climate, name='daily_climate'),
    path('data-visualization/customize/', customize_visualization, name="customize_visualization"),
    path('data-visualization/gridspec/', weather_phenomena, name='gridspec'),
    path('machine_learning/supervised/knn/', sklearn_knn, name='sklearn_knn'),
]