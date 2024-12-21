from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.chat, name='chat'),
    path('create_personality/', views.create_personality, name='create_personality'),
    path('create_personality_epub/', views.create_personality_epub, name='create_personality_epub'),
    path('get_personality_settings/<str:personality_name>/', views.get_personality_settings, name='get_personality_settings'),
    path('adjust_personality/', views.adjust_personality, name='adjust_personality'),
    path('delete_personality/', views.delete_personality, name='delete_personality'),
    path('personality_manager/', views.personality_manager, name='personality_manager'),
    path('chat_between_personalities/', views.chat_between_personalities, name='chat_between_personalities'),  
]