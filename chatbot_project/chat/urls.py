from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.chat, name='chat'),
    path('create_personality/', views.create_personality, name='create_personality'),
    path('create_personality_epub/', views.create_personality_epub, name='create_personality_epub'),
]