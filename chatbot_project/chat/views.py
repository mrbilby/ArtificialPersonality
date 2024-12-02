
# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .chatbot import PersonalityManager, ChatBot


class ChatbotManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.personality_manager = PersonalityManager()
        self.active_chatbots = {}
    
    def get_or_create_chatbot(self, personality_name):
        if personality_name not in self.active_chatbots:
            personality, short_memory, long_memory = (
                self.personality_manager.load_or_create_personality(personality_name)
            )
            self.active_chatbots[personality_name] = ChatBot(
                personality, short_memory, long_memory
            )
        return self.active_chatbots[personality_name]
    
    def save_all_chatbots(self):
        for name, chatbot in self.active_chatbots.items():
            self.personality_manager.save_personality_state(
                name,
                chatbot.personality,
                chatbot.short_memory,
                chatbot.long_memory,
                chatbot.memory_consolidator
            )

def index(request):
    chatbot_manager = ChatbotManager.get_instance()
    personalities = chatbot_manager.personality_manager.list_available_personalities()
    print(f"[Debug] Available personalities: {personalities}")  # Add this debug line
    print(f"[Debug] Looking for personalities in: {chatbot_manager.personality_manager.personality_path}")  # Add this
    return render(request, 'chat/index.html', {'personalities': personalities})

# chat/views.py - modify the chat view to handle the bye command
# chat/views.py

@csrf_exempt
def chat(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message')
            personality_name = data.get('personality', 'default')
            is_bye = data.get('is_bye', False)
            
            chatbot_manager = ChatbotManager.get_instance()
            chatbot = chatbot_manager.get_or_create_chatbot(personality_name)
            
            response = chatbot.process_query(message)
            
            if is_bye or message.lower() == 'bye':
                chatbot_manager.save_all_chatbots()
                return JsonResponse({
                    'response': response,
                    'terminated': True
                })
                
            return JsonResponse({'response': response})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    # Handle GET request
    elif request.method == 'GET':
        personality_name = request.GET.get('personality', 'default')
        return render(request, 'chat/chat.html', {'personality_name': personality_name})

    # Handle other methods
    return HttpResponse(status=405)  # Method not allowed