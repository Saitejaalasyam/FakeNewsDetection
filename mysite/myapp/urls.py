from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('next', views.Next, name='Next'),
    path('result',views.result,name='Result')
]