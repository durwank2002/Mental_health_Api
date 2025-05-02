from django.urls import path
from .views import MentalHealthPrediction

urlpatterns = [
    path('predict/', MentalHealthPrediction.as_view(), name='predict'),
]
