from django.contrib import admin
from django.urls import path
from detector import views  # <-- 1. Import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.predict_tweet, name='predict_tweet'),  # <-- 2. Home URL സെറ്റ് ചെയ്യുക
]