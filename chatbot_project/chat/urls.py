from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.chat, name='chat'),
    path('create_personality/', views.create_personality, name='create_personality'),
    path('create_from_epub/', views.create_from_epub, name='create_from_epub'),
]