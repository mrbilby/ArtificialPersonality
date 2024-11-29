import json
import tempfile
import os
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup
from openai import OpenAI
from typing import Dict, List, Tuple
import networkx as nx
from datetime import datetime, timedelta
from dotenv import load_dotenv
import tiktoken

load_dotenv()

class CharacterPersonalityExtractor:
    def __init__(self, openai_api_key: str, token_limit: int = 60000):
        self.client = OpenAI(api_key=openai_api_key)
        self.token_limit = token_limit
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def truncate_text_to_token_limit(self, text: str, limit: int) -> str:
        """Truncate text to specified token limit."""
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= limit:
            return text
        return encoding.decode(tokens[:limit])
        
    def epub_to_text(self, epub_path: str) -> str:
        """Convert epub to plain text and truncate to token limit."""
        book = epub.read_epub(epub_path)
        chapters = []
        total_text = ""
        
        # Process chapters one at a time to manage token count
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                chapter_text = soup.get_text()
                total_text += chapter_text + "\n\n"
                
                # Check token count and truncate if necessary
                if self.count_tokens(total_text) > self.token_limit:
                    print(f"Truncating text to {self.token_limit} tokens...")
                    return self.truncate_text_to_token_limit(total_text, self.token_limit)
                
        return total_text

    def generate_personality(self, character_analysis: str) -> Dict:
        """Generate personality.json content based on character analysis."""
        prompt = f"""
        Based on this character analysis, create a personality profile with the following structure:
        {{
            "tone": "(how the character typically communicates)",
            "response_style": "(their typical way of engaging with others)",
            "behavior": "(their behavioral patterns)",
            "user_preferences": {{
                "likes": [],
                "dislikes": [],
                "preferences": {{}}
            }},
            "do_dont": {{
                "do": [],
                "dont": []
            }}
        }}

        Return only the JSON object, properly formatted. Ensure all values are properly quoted strings,
        arrays, or objects. The response must be valid JSON.

        Character Analysis:
        {character_analysis}
        """

        # Calculate available tokens for the prompt
        template_tokens = self.count_tokens(prompt)
        available_tokens = min(self.token_limit, 4096 - template_tokens - 100)
        truncated_analysis = self.truncate_text_to_token_limit(character_analysis, available_tokens)
        
        final_prompt = prompt.replace(character_analysis, truncated_analysis)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a character analyst creating personality profiles. Always return valid JSON."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON if it's wrapped in other text
            try:
                # First try to parse as-is
                personality_data = json.loads(response_text)
            except json.JSONDecodeError:
                # If that fails, try to find JSON-like structure
                import re
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    personality_data = json.loads(json_match.group(1))
                else:
                    raise ValueError("Could not extract valid JSON from response")

            # Ensure all required fields are present
            required_fields = ['tone', 'response_style', 'behavior', 'user_preferences', 'do_dont']
            for field in required_fields:
                if field not in personality_data:
                    personality_data[field] = ""

            return personality_data

        except Exception as e:
            print(f"Error in generate_personality: {str(e)}")
            print(f"Raw response: {response_text if 'response_text' in locals() else 'No response received'}")
            
            # Return a default personality if generation fails
            return {
                "tone": "neutral",
                "response_style": "direct",
                "behavior": "standard",
                "user_preferences": {
                    "likes": [],
                    "dislikes": [],
                    "preferences": {}
                },
                "do_dont": {
                    "do": [],
                    "dont": []
                }
            }

    def extract_memories(self, epub_content: str, character_name: str) -> List[Dict]:
        """Extract significant memories/interactions involving the character."""
        prompt = f"""
        Extract significant memories and interactions involving {character_name} from the text.
        Format the response as a JSON array of objects, where each object has the following structure:
        [
            {{
                "description": "brief description of the event",
                "characters": ["list", "of", "involved", "characters"],
                "emotional_impact": "description of emotional impact",
                "tags": ["relevant", "tags", "or", "themes"],
                "timing": "early/middle/late"
            }}
        ]

        Return only the JSON array, properly formatted. Ensure all values are properly quoted strings or arrays.
        
        Text content:
        {epub_content}
        """
        
        # Calculate available tokens for the content
        template_tokens = self.count_tokens(prompt)
        available_tokens = min(self.token_limit, 4096 - template_tokens - 100)
        truncated_content = self.truncate_text_to_token_limit(epub_content, available_tokens)
        
        final_prompt = prompt.replace(epub_content, truncated_content)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a literary analyst extracting character memories and interactions. Always return valid JSON."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            response_text = response.choices[0].message.content.strip()
            
            try:
                memories = json.loads(response_text)
                if not isinstance(memories, list):
                    memories = [memories]
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'(\[[\s\S]*\])', response_text)
                if json_match:
                    memories = json.loads(json_match.group(1))
                else:
                    raise ValueError("Could not extract valid JSON from response")

            return memories

        except Exception as e:
            print(f"Error in extract_memories: {str(e)}")
            print(f"Raw response: {response_text if 'response_text' in locals() else 'No response received'}")
            
            # Return a default memory if extraction fails
            return [{
                "description": "Default memory created due to extraction error",
                "characters": [character_name],
                "emotional_impact": "neutral",
                "tags": ["error"],
                "timing": "early"
            }]

    def create_memory_graph(self, memories: List[Dict]) -> nx.Graph:
        """Create a memory graph from extracted memories."""
        G = nx.Graph()
        
        # Add nodes for each memory
        for i, memory in enumerate(memories):
            G.add_node(i, 
                      description=memory['description'],
                      characters=memory['characters'],
                      emotional_impact=memory['emotional_impact'],
                      tags=memory['tags'],
                      timing=memory['timing'])
        
        # Create edges between related memories
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                # Calculate similarity based on shared tags and characters
                shared_tags = len(set(memories[i]['tags']) & set(memories[j]['tags']))
                shared_chars = len(set(memories[i]['characters']) & set(memories[j]['characters']))
                
                similarity = (shared_tags * 0.6) + (shared_chars * 0.4)
                if similarity > 0:
                    G.add_edge(i, j, weight=similarity)
        
        return G

    def format_long_term_memory(self, memories: List[Dict]) -> List[Dict]:
        """Format memories for long_term_memory.json."""
        formatted_memories = []
        base_time = datetime.now() - timedelta(days=365)  # Start from a year ago
        
        for i, memory in enumerate(memories):
            # Calculate relative timestamp based on story timing
            if memory['timing'] == 'early':
                time_offset = timedelta(days=i*30)  # Space memories month apart
            elif memory['timing'] == 'middle':
                time_offset = timedelta(days=180 + i*30)
            else:  # 'late'
                time_offset = timedelta(days=270 + i*30)
                
            timestamp = base_time + time_offset
            
            formatted_memories.append({
                'user_message': memory['description'],
                'bot_response': f"I remember this event clearly. {memory['emotional_impact']}",
                'tags': memory['tags'],
                'timestamp': timestamp.isoformat(),
                'priority_score': 0.8 if 'significant' in memory['tags'] else 0.5,
                'priority_factors': {
                    'emotional_impact': 0.8 if 'emotional' in memory['tags'] else 0.5,
                    'recency': 0.7,
                    'user_importance': 0.6
                }
            })
        
        return formatted_memories

    def _get_character_analysis(self, epub_content: str, character_name: str) -> str:
        """Get detailed character analysis with token management."""
        prompt = f"""
        Provide a detailed analysis of {character_name} from the text, including:
        1. Personality traits and characteristics
        2. Communication style and typical behaviors
        3. Values and beliefs
        4. Relationships and interactions with others
        5. Character development and growth

        Text content:
        """
        
        # Calculate available tokens for the content
        template_tokens = self.count_tokens(prompt)
        available_tokens = min(self.token_limit, 4096 - template_tokens - 100)
        truncated_content = self.truncate_text_to_token_limit(epub_content, available_tokens)
        
        final_prompt = prompt + truncated_content

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a literary analyst specialized in character analysis."},
                {"role": "user", "content": final_prompt}
            ]
        )

        return response.choices[0].message.content

    def process_epub(self, epub_path: str, character_name: str) -> Tuple[Dict, List[Dict], nx.Graph]:
        """Process epub file and generate all required outputs."""
        # Extract text content with token limit
        print("Extracting text from epub...")
        epub_content = self.epub_to_text(epub_path)
        token_count = self.count_tokens(epub_content)
        print(f"Extracted text token count: {token_count}")
        
        # Get character analysis with token limit
        print("Analyzing character...")
        character_analysis = self._get_character_analysis(epub_content, character_name)
        
        # Generate personality profile
        print("Generating personality profile...")
        personality = self.generate_personality(character_analysis)
        personality['name'] = character_name
        
        # Extract and process memories
        print("Extracting memories...")
        memories = self.extract_memories(epub_content, character_name)
        long_term_memories = self.format_long_term_memory(memories)
        memory_graph = self.create_memory_graph(memories)
        
        return personality, long_term_memories, memory_graph


def main():
    # Get inputs
    epub_path = input("Enter the path to your epub file: ")
    character_name = input("Enter the character name to extract: ")
    token_limit = 60000  # Set default token limit
    
    api_key = os.getenv("API_KEY")
    
    if not api_key:
        print("Error: API_KEY not found in environment variables")
        return
    
    try:
        # Initialize extractor with token limit
        extractor = CharacterPersonalityExtractor(api_key, token_limit)
        
        # Process epub
        print(f"Processing epub with {token_limit} token limit...")
        personality, long_term_memories, memory_graph = extractor.process_epub(epub_path, character_name)
        
        # Save personality.json
        with open(f"{character_name.lower()}_personality.json", 'w') as f:
            json.dump(personality, f, indent=4)
        
        # Save long_term_memory.json
        with open(f"{character_name.lower()}_long_term_memory.json", 'w') as f:
            json.dump(long_term_memories, f, indent=4)
        
        # Save memory_graph.json
        graph_data = {
            'nodes': [[n, data] for n, data in memory_graph.nodes(data=True)],
            'edges': [[u, v, data] for u, v, data in memory_graph.edges(data=True)]
        }
        with open(f"{character_name.lower()}_memory_graph.json", 'w') as f:
            json.dump(graph_data, f, indent=4)
            
        print(f"Successfully created personality and memory files for {character_name}!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()