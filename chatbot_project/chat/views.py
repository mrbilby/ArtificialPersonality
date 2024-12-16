
# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .chatbot import (
    PersonalityManager, 
    ChatBot, 
    PersonalityProfile,
    ShortTermMemory,
    LongTermMemory,
    MemoryConsolidator,
    ChatbotManager,
)
import io
import sys
from contextlib import redirect_stdout
from io import StringIO

class OutputCapture:
    def __init__(self):
        self.outputs = []
        self.original_stdout = sys.stdout
        self.capture_buffer = StringIO()
    
    def start(self):
        sys.stdout = self.capture_buffer
    
    def stop(self):
        sys.stdout = self.original_stdout
        output = self.capture_buffer.getvalue()
        self.capture_buffer = StringIO()
        return output

output_capture = OutputCapture()



def index(request):
    chatbot_manager = ChatbotManager.get_instance()
    personalities = chatbot_manager.personality_manager.list_available_personalities()
    print(f"[Debug] Available personalities: {personalities}")  # Add this debug line
    print(f"[Debug] Looking for personalities in: {chatbot_manager.personality_manager.personality_path}")  # Add this
    return render(request, 'chat/index.html', {'personalities': personalities})

# chat/views.py - modify the chat view to handle the bye command
# chat/views.py

# In views.py
@csrf_exempt
def chat(request):
    if request.method == 'POST':
        try:
            stdout_capture = StringIO()
            with redirect_stdout(stdout_capture):
                print("TESTING DIAGNOSTIC OUTPUT - IF YOU SEE THIS, CAPTURE WORKS")

                data = json.loads(request.body)
                message = data.get('message', '')
                personality_name = data.get('personality', 'default')
                is_bye = data.get('is_bye', False)

                # This is the new field we expect from the updated frontend logic.
                # It's a dictionary of filename: content pairs.
                file_contents = data.get('file_contents', {})

                # If there are uploaded files, append their contents to the message
                if file_contents:
                    # Add a section to the message with all file contents
                    message += "\n\nUploaded File Contents:\n"
                    for fname, fcontent in file_contents.items():
                        message += f"Filename: {fname}\n{fcontent}\n\n"

                # Initialize and get the chatbot instance
                chatbot_manager = ChatbotManager.get_instance()
                chatbot = chatbot_manager.get_or_create_chatbot(personality_name)

                # Process the user's query with the possibly augmented message
                response = chatbot.process_query(message)

                diagnostic_output = stdout_capture.getvalue()
                
                if not data.get('diagnostic_mode', False):
                    diagnostic_output = None

                return JsonResponse({
                    'response': response,
                    'diagnostic_output': diagnostic_output,
                    'terminated': is_bye or (message.strip().lower() == 'bye')
                })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    elif request.method == 'GET':
        personality_name = request.GET.get('personality', 'default')
        return render(request, 'chat/chat.html', {'personality_name': personality_name})
    
    return HttpResponse(status=405)

def create_personality(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        chatbot_manager = ChatbotManager.get_instance()
        personality_name = data['name'].lower()
        
        try:
            # Create personality profile
            personality = PersonalityProfile(
                tone=data.get('tone', 'neutral'),
                response_style=data.get('response_style', 'detailed'),
                behavior=data.get('behavior', 'reactive'),
                user_preferences=data.get('user_preferences', {}),
                name=personality_name
            )
            
            # Add do/don't rules
            for do_rule in data['do_dont'].get('do', []):
                personality.add_do_rule(do_rule)
            for dont_rule in data['do_dont'].get('dont', []):
                personality.add_dont_rule(dont_rule)
            
            # Initialize memory components
            short_memory = ShortTermMemory(max_interactions=25)
            long_memory = LongTermMemory(max_interactions=1000, personality_name=personality_name)
            
            # Get file paths
            personality_manager = chatbot_manager.personality_manager
            files = personality_manager.get_personality_files(personality_name)
            
            # Save all components
            personality.save_to_file(files["personality"])
            
            # Save memory files with initial empty states
            with open(files["short_memory"], "w") as f:
                json.dump([], f)
            with open(files["long_memory"], "w") as f:
                json.dump([], f)
            
            # Initialize and save consolidated memory
            memory_consolidator = MemoryConsolidator(short_memory, long_memory, personality_name)
            memory_consolidator.save_consolidated_memory(files["consolidated_memory"])
            
            print(f"[Debug] Created and saved all files for personality: {personality_name}")
            print(f"[Debug] Short-term memory file: {files['short_memory']}")
            print(f"[Debug] Long-term memory file: {files['long_memory']}")
            
            return JsonResponse({'success': True})
            
        except Exception as e:
            print(f"[Error] Failed to create personality: {e}")
            return JsonResponse({'error': str(e)}, status=500)
            
    return render(request, 'chat/create_personality.html')

def create_from_epub(request):
    return render(request, 'chat/create_from_epub.html')