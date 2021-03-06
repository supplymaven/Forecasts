"""Forecasts URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.contrib.auth import views as auth_views
from MachineLearning import views
from django.urls import path, re_path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('selections/', views.selections, name='selections'),
    path('index/', views.index, name='index'),
    path('', views.home, name='home'),
    path('arima/', views.arima, name='arima'),
    re_path(r'^forecast-model/(?P<series>[-\w]+)/(?P<model>[\w]+)/$', views.forecast_model, name='forecast_model'),
]
