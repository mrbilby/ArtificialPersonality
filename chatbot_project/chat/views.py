
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
from .character_extracter_personality import CharacterPersonalityExtractor
import tempfile
import os
import shutil


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

                chatbot_manager = ChatbotManager.get_instance()
                chatbot = chatbot_manager.get_or_create_chatbot(personality_name)

                # Handle "bye" command
                if is_bye or (message.strip().lower() == 'bye'):
                    return JsonResponse({
                        'response': 'Goodbye! The chat session has been terminated.',
                        'terminated': True,  # Signal the frontend to terminate the chat
                    })

                # Handle thinking mode
                if thinking_mode and thinking_minutes > 0:
                    # Process thinking asynchronously
                    async def run_thinking():
                        return await process_thinking(
                            message, file_contents, thinking_minutes, personality_name, chatbot.client
                        )

                    success, result, file_path = asyncio.run(run_thinking())
                    diagnostic_output = stdout_capture.getvalue()

                    if not diagnostic_mode:
                        diagnostic_output = None

                    if success:
                        return JsonResponse({
                            'response': result,
                            'file_path': file_path,  # Include file path in response
                            'thinking_complete': True,
                            'diagnostic_output': diagnostic_output
                        })
                    else:
                        return JsonResponse({
                            'error': f"Error during thinking process: {result}",
                            'diagnostic_output': diagnostic_output
                        })

                # Handle regular message processing
                if file_contents:
                    file_summary = summarize_files(file_contents, chatbot.client)
                    if file_summary:
                        message += f"\n\nFile Contents Summary:\n{file_summary}"

                response = chatbot.process_query(message)
                diagnostic_output = stdout_capture.getvalue()

                if not diagnostic_mode:
                    diagnostic_output = None

                return JsonResponse({
                    'response': response,
                    'diagnostic_output': diagnostic_output,
                    'terminated': False  # Chat continues
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
    thoughts_file = f"thoughts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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

        with open(thoughts_file, 'w', encoding='utf-8') as f:
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

                # Save to file
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"=== Thought {i + 1}/{iterations} ===\n")
                f.write(f"Time: {timestamp}\n")
                f.write(response)
                f.write("\n\n")
                f.flush()

                # Append to final output
                final_output += f"=== Thought {i + 1}/{iterations} ===\nTime: {timestamp}\n{response}\n\n"

                # Store thought for next iteration
                previous_thoughts.append(f"Thought {i + 1}: {response}")

                # Wait before next thought (unless it's the last one)
                if i < iterations - 1:
                    await asyncio.sleep(60)

        # Return the collected thoughts and saved file path
        return True, final_output, thoughts_file

    except Exception as e:
        print(f"Error in thinking process: {e}")
        return False, str(e), None


        
def create_personality_epub(request):
    if request.method == 'POST':
        try:
            # Get form data
            character_name = request.POST.get('characterName')
            epub_file = request.FILES.get('epubFile')
            
            if not character_name or not epub_file:
                return JsonResponse({
                    'error': 'Both character name and EPUB file are required'
                }, status=400)

            # Get API key from environment
            api_key = os.getenv("API_KEY")
            if not api_key:
                return JsonResponse({
                    'error': 'API key not configured'
                }, status=500)

            # Set up paths relative to views.py location
            base_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_path, 'data')
            personalities_path = os.path.join(data_path, 'personalities')
            memories_path = os.path.join(data_path, 'memories')
            
            # Create directories if they don't exist
            os.makedirs(personalities_path, exist_ok=True)
            os.makedirs(memories_path, exist_ok=True)

            # Initialize the extractor
            extractor = CharacterPersonalityExtractor(api_key)

            # Create a temporary file to store the uploaded EPUB
            with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as temp_epub:
                for chunk in epub_file.chunks():
                    temp_epub.write(chunk)
                temp_epub_path = temp_epub.name

            try:
                # Process the EPUB file
                print(f"Processing EPUB for character: {character_name}")
                personality = extractor.process_epub(temp_epub_path, character_name)

                # Create personality profile
                profile = PersonalityProfile(
                    tone=personality.get("tone", "neutral"),
                    response_style=personality.get("response_style", "detailed"),
                    behavior=personality.get("behavior", "reactive"),
                    user_preferences=personality.get("user_preferences", {}),
                    name=character_name.lower()
                )

                # Add do/don't rules
                for do_rule in personality.get("do_dont", {}).get("do", []):
                    profile.add_do_rule(do_rule)
                for dont_rule in personality.get("do_dont", {}).get("dont", []):
                    profile.add_dont_rule(dont_rule)

                # Initialize memory components
                short_memory = ShortTermMemory(max_interactions=25)
                long_memory = LongTermMemory(max_interactions=1000, personality_name=character_name.lower())

                # Define file paths
                personality_file = os.path.join(personalities_path, f"{character_name.lower()}_personality.json")
                short_memory_file = os.path.join(memories_path, f"{character_name.lower()}_short_term_memory.json")
                long_memory_file = os.path.join(memories_path, f"{character_name.lower()}_long_term_memory.json")
                consolidated_memory_file = os.path.join(memories_path, f"{character_name.lower()}_consolidated_memory.json")

                # Save files
                profile.save_to_file(personality_file)
                print(f"Saved personality file to: {personality_file}")

                with open(short_memory_file, "w") as f:
                    json.dump([], f)
                print(f"Saved short-term memory to: {short_memory_file}")

                with open(long_memory_file, "w") as f:
                    json.dump([], f)
                print(f"Saved long-term memory to: {long_memory_file}")

                # Initialize and save consolidated memory
                memory_consolidator = MemoryConsolidator(short_memory, long_memory, character_name.lower())
                memory_consolidator.save_consolidated_memory(file_path=consolidated_memory_file)
                print(f"Saved consolidated memory to: {consolidated_memory_file}")

                return JsonResponse({'success': True})

            finally:
                # Clean up the temporary file
                if os.path.exists(temp_epub_path):
                    os.unlink(temp_epub_path)

        except Exception as e:
            print(f"Error processing EPUB: {str(e)}")
            return JsonResponse({
                'error': f'Failed to process EPUB: {str(e)}'
            }, status=500)

    return render(request, 'chat/create_personality_epub.html')

@csrf_exempt
def adjust_personality(request):
    """Update personality settings with support for flexible JSON structures"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        personality_name = data.get('name')
        
        if not personality_name:
            return JsonResponse({'error': 'Personality name is required'}, status=400)

        chatbot_manager = ChatbotManager.get_instance()
        personality, short_memory, long_memory = chatbot_manager.personality_manager.load_or_create_personality(personality_name)
        
        # Get the existing personality data to preserve structure
        personality_manager = chatbot_manager.personality_manager
        files = personality_manager.get_personality_files(personality_name)
        
        try:
            with open(files["personality"], 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}

        # Update the personality data while preserving structure
        def update_dict_recursively(existing, new):
            """Recursively update dictionary while preserving structure"""
            if not isinstance(existing, dict) or not isinstance(new, dict):
                return new
            
            result = existing.copy()
            for key, value in new.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = update_dict_recursively(result[key], value)
                else:
                    result[key] = value
            return result

        # Update core personality attributes if they exist in the request
        for attr in ['tone', 'response_style', 'behavior']:
            if attr in data:
                existing_data[attr] = data[attr]

        # Update nested structures
        for key in ['user_preferences', 'do_dont', 'personality_traits', 
                   'background_influence', 'social_dynamics']:
            if key in data:
                if key not in existing_data:
                    existing_data[key] = {}
                existing_data[key] = update_dict_recursively(existing_data[key], data[key])

        # Ensure name is preserved
        existing_data['name'] = personality_name

        # Save the updated personality file
        with open(files["personality"], 'w') as f:
            json.dump(existing_data, f, indent=4)

        # Update the personality object with core attributes
        personality.tone = existing_data.get('tone', '')
        personality.response_style = existing_data.get('response_style', '')
        personality.behavior = existing_data.get('behavior', '')
        personality.user_preferences = existing_data.get('user_preferences', {})
        personality.do_dont = existing_data.get('do_dont', {'do': [], 'dont': []})

        print(f"[Debug] Successfully updated personality: {personality_name}")
        return JsonResponse({'success': True})
        
    except Exception as e:
        print(f"[Error] Failed to update personality: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def get_personality_settings(request, personality_name):
    """Fetch current settings for a personality with full structure"""
    try:
        chatbot_manager = ChatbotManager.get_instance()
        personality_manager = chatbot_manager.personality_manager
        files = personality_manager.get_personality_files(personality_name)
        
        try:
            with open(files["personality"], 'r') as f:
                personality_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            personality_data = {
                'tone': '',
                'response_style': '',
                'behavior': '',
                'user_preferences': {},
                'do_dont': {'do': [], 'dont': []},
                'name': personality_name
            }
        
        return JsonResponse(personality_data)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def delete_personality(request):
    """Delete a personality and its related files"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        personality_name = data['personality']
        
        # Don't allow deletion of default personality
        if personality_name.lower() == 'default':
            return JsonResponse({'error': 'Cannot delete default personality'}, status=400)
        
        chatbot_manager = ChatbotManager.get_instance()
        personality_manager = chatbot_manager.personality_manager
        
        # Get all related file paths
        files = personality_manager.get_personality_files(personality_name)
        
        # Remove the personality from active chatbots if it exists
        if personality_name in chatbot_manager.active_chatbots:
            del chatbot_manager.active_chatbots[personality_name]
        
        # Delete all related files
        for file_path in files.values():
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[Debug] Deleted file: {file_path}")
            except Exception as e:
                print(f"[Warning] Failed to delete file {file_path}: {e}")
        
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def personality_manager(request):
    chatbot_manager = ChatbotManager.get_instance()
    personalities = chatbot_manager.personality_manager.list_available_personalities()
    print(f"[Debug] Available personalities: {personalities}")  # Add this debug line
    print(f"[Debug] Looking for personalities in: {chatbot_manager.personality_manager.personality_path}")  # Add this
    return render(request, 'chat/personality_manager.html', {'personalities': personalities})


def chat_between_personalities(request):
    """View for handling chat between personalities, without memory storage."""
    if request.method == 'GET':
        chatbot_manager = ChatbotManager.get_instance()
        personalities = chatbot_manager.personality_manager.list_available_personalities()
        return render(request, 'chat/chat_between_personalities.html', {
            'personalities': personalities
        })
    
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            topic = data.get('topic', '')
            speaking_personality = data.get('speaking_personality', '')
            listening_personality = data.get('listening_personality', '')
            conversation_history = data.get('conversation', [])
            is_user_interjection = data.get('is_user_interjection', False)

            if not all([topic, speaking_personality, listening_personality]):
                return JsonResponse({'error': 'Missing required parameters'}, status=400)

            # Get chatbot instances
            chatbot_manager = ChatbotManager.get_instance()
            speaking_chatbot = chatbot_manager.get_or_create_chatbot(speaking_personality)

            # Build context prompt
            if is_user_interjection:
                system_prompt = f"""You are {speaking_personality} participating in a group conversation about {topic}. 
A user has just joined the conversation and addressed you. Stay in character while acknowledging their input.

Your personality:
{speaking_chatbot.personality.tone}
{speaking_chatbot.personality.response_style}
{speaking_chatbot.personality.behavior}

Previous conversation:"""
            else:
                system_prompt = f"""You are {speaking_personality} having a conversation with {listening_personality} about {topic}.

Your personality:
{speaking_chatbot.personality.tone}
{speaking_chatbot.personality.response_style}
{speaking_chatbot.personality.behavior}

Previous conversation:"""

            # Add conversation history
            for entry in conversation_history:
                system_prompt += f"\n{entry['personality']}: {entry['message']}"

            # Generate response using the chatbot's client
            response = speaking_chatbot.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate your next response in the conversation:"}
                ],
                max_tokens=500,
                temperature=0.7
            )

            response_text = response.choices[0].message.content.strip()

            return JsonResponse({
                'response': response_text,
                'speaking_personality': speaking_personality,
                'listening_personality': listening_personality
            })

        except Exception as e:
            print(f"[Error] Failed to generate chat response: {e}")
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)