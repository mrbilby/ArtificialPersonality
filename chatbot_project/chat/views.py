
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
import asyncio
from datetime import datetime

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

                # Extract data from the request
                data = json.loads(request.body)
                message = data.get('message', '')
                personality_name = data.get('personality', 'default')
                is_bye = data.get('is_bye', False)
                file_contents = data.get('file_contents', {})
                thinking_mode = data.get('thinking_mode', False)
                thinking_minutes = data.get('thinking_minutes', 0)
                diagnostic_mode = data.get('diagnostic_mode', False)

                # Initialize chatbot manager and chatbot instance
                chatbot_manager = ChatbotManager.get_instance()
                chatbot = chatbot_manager.get_or_create_chatbot(personality_name)

                if thinking_mode and thinking_minutes > 0:
                    # Process thinking asynchronously
                    async def run_thinking():
                        return await process_thinking(
                            message, file_contents, thinking_minutes, personality_name, chatbot.client
                        )

                    # Execute the asynchronous function
                    success, result = asyncio.run(run_thinking())
                    diagnostic_output = stdout_capture.getvalue()

                    if not diagnostic_mode:
                        diagnostic_output = None

                    if success:
                        # Return final thinking output
                        return JsonResponse({
                            'response': result,
                            'thinking_complete': True,
                            'diagnostic_output': diagnostic_output
                        })
                    else:
                        # Return error during thinking process
                        return JsonResponse({
                            'error': f"Error during thinking process: {result}",
                            'diagnostic_output': diagnostic_output
                        })

                # Regular message processing
                if file_contents:
                    # Summarize file contents if provided
                    file_summary = summarize_files(file_contents, chatbot.client)
                    if file_summary:
                        message += f"\n\nFile Contents Summary:\n{file_summary}"

                # Process the message through the chatbot
                response = chatbot.process_query(message)
                diagnostic_output = stdout_capture.getvalue()

                if not diagnostic_mode:
                    diagnostic_output = None

                return JsonResponse({
                    'response': response,
                    'diagnostic_output': diagnostic_output,
                    'terminated': is_bye or (message.strip().lower() == 'bye')
                })

        except Exception as e:
            # Catch and return any server-side errors
            return JsonResponse({'error': str(e)}, status=500)

    elif request.method == 'GET':
        # Render the chat interface for GET requests
        personality_name = request.GET.get('personality', 'default')
        return render(request, 'chat/chat.html', {'personality_name': personality_name})

    # Return a 405 for unsupported request methods
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

def summarize_files(file_contents: dict, client) -> str:
    """
    Summarize the contents of uploaded files using the model.
    Returns a synopsis of the files' contents.
    """
    if not file_contents:
        return ""
    
    # Create a prompt for summarization
    files_text = "\n\n".join([
        f"File: {fname}\nContents:\n{content}"
        for fname, content in file_contents.items()
    ])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Provide a concise synopsis of the following file(s). Focus on key points and main content. Start with 'Files Synopsis:' followed by your summary."
                },
                {
                    "role": "user",
                    "content": files_text
                }
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Error] Failed to summarize files: {e}")
        return "Error: Could not generate file summary."

async def process_thinking(message, file_contents, minutes, personality_name, client):
    """Process thinking iterations with proper delays and context building."""
    final_output = ""  # Collect all thoughts here

    try:
        chatbot_manager = ChatbotManager.get_instance()
        chatbot = chatbot_manager.get_or_create_chatbot(personality_name)
        
        # Initialize context with base message and any file contents
        running_context = message
        if file_contents:
            file_summary = summarize_files(file_contents, client)
            running_context += f"\n\nContext from files:\n{file_summary}"
        
        previous_thoughts = []
        iterations = int(minutes)
        
        for i in range(iterations):
            if previous_thoughts:
                thinking_prompt = (
                    f"Previous thoughts:\n"
                    f"{chr(10).join(previous_thoughts)}\n\n"
                    f"Building upon these previous thoughts (iteration {i + 1}/{iterations}):\n"
                    f"Continue developing ideas about: {running_context}"
                )
            else:
                thinking_prompt = (
                    f"Initial thinking iteration {i + 1}/{iterations}:\n"
                    f"Begin deep analysis of: {running_context}"
                )
            
            # Generate new thought
            response = chatbot.process_query(thinking_prompt)
            
            # Append to final output
            final_output += f"=== Thought {i + 1}/{iterations} ===\n{response}\n\n"

            # Store thought for the next iteration
            previous_thoughts.append(f"Thought {i + 1}: {response}")
            
            # Wait before the next iteration (unless it's the last one)
            if i < iterations - 1:
                await asyncio.sleep(30)

        # Return the collected thoughts
        return True, final_output

    except Exception as e:
        print(f"Error in thinking process: {e}")
        return False, str(e)



        
def create_from_epub(request):
    return render(request, 'chat/create_from_epub.html')