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
        Based on this character analysis, create a detailed personality profile with the following structure.
        Provide extensive details for each category:
        {{
            "tone": "Detailed description of communication style, speech patterns, and typical expressions",
            "response_style": "In-depth analysis of how they interact with others, including conversation patterns",
            "behavior": "Comprehensive overview of behavioral patterns, reactions, and decision-making style",
            "user_preferences": {{
                "likes": [
                    "At least 10 specific things they enjoy, value, or appreciate",
                    "Include hobbies, attitudes, and preferences"
                ],
                "dislikes": [
                    "At least 10 specific things they dislike or avoid",
                    "Include pet peeves, frustrations, and sources of discomfort"
                ],
                "preferences": {{
                    "social_style": "How they prefer to interact socially",
                    "communication": "Preferred communication methods and styles",
                    "environment": "Preferred surroundings and conditions",
                    "relationships": "How they approach different types of relationships",
                    "daily_routine": "Preferred daily patterns and habits",
                    "conflict_resolution": "How they handle disagreements and problems",
                    "decision_making": "Their approach to making choices",
                    "leisure": "How they prefer to spend free time",
                    "work_style": "How they approach tasks and responsibilities",
                    "emotional_expression": "How they express and handle emotions"
                }}
            }},
            "do_dont": {{
                "do": [
                    "At least 10 specific behaviors and actions they consistently demonstrate",
                    "Include moral principles, habits, and characteristic behaviors"
                ],
                "dont": [
                    "At least 10 specific behaviors and actions they avoid or resist",
                    "Include personal rules, boundaries, and things they refuse to do"
                ]
            }},
            "personality_traits": {{
                "strengths": ["List at least 5 major character strengths"],
                "weaknesses": ["List at least 5 major character flaws or challenges"],
                "growth_areas": ["List 3-5 areas where character shows development"],
                "core_values": ["List 5-7 fundamental values that drive the character"],
                "coping_mechanisms": ["List 3-5 ways they handle stress or difficulties"]
            }},
            "background_influence": {{
                "key_experiences": ["List 3-5 formative experiences that shaped them"],
                "relationships": ["List significant relationships and their impact"],
                "worldview": "Description of how they see the world and their place in it"
            }},
            "social_dynamics": {{
                "leadership_style": "How they handle leadership or authority",
                "group_role": "Their typical role in group situations",
                "friendship_approach": "How they build and maintain friendships",
                "trust_patterns": "How they develop and maintain trust",
                "conflict_style": "How they handle confrontation and disagreement"
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
                    {
                        "role": "system", 
                        "content": "You are a character analyst creating personality profiles. Return only the JSON object with no formatting tokens, no ```json markers, and no backticks."
                    },
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            response_text = response.choices[0].message.content.strip()
            
            # Remove any markdown formatting
            response_text = response_text.replace('```json', '').replace('```', '')
            
            try:
                personality_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {str(e)}")
                print("Raw response:", response_text)
                # If parsing fails, return the default structure
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
                    },
                    "personality_traits": {
                        "strengths": [],
                        "weaknesses": [],
                        "growth_areas": [],
                        "core_values": [],
                        "coping_mechanisms": []
                    },
                    "background_influence": {
                        "key_experiences": [],
                        "relationships": [],
                        "worldview": ""
                    },
                    "social_dynamics": {
                        "leadership_style": "",
                        "group_role": "",
                        "friendship_approach": "",
                        "trust_patterns": "",
                        "conflict_style": ""
                    }
                }

            return personality_data

        except Exception as e:
            print(f"API call failed: {str(e)}")
            # Return default structure on API failure
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
                },
                "personality_traits": {
                    "strengths": [],
                    "weaknesses": [],
                    "growth_areas": [],
                    "core_values": [],
                    "coping_mechanisms": []
                },
                "background_influence": {
                    "key_experiences": [],
                    "relationships": [],
                    "worldview": ""
                },
                "social_dynamics": {
                    "leadership_style": "",
                    "group_role": "",
                    "friendship_approach": "",
                    "trust_patterns": "",
                    "conflict_style": ""
                }
            }

    def extract_memories(self, epub_content: str, character_name: str) -> List[Dict]:
        """Extract significant memories/interactions involving the character."""
        base_prompt = f"""
        You must respond with ONLY a JSON array of memory objects for {character_name}.
        Each object must follow this exact structure:
        {{
            "description": "string describing the event",
            "characters": ["array of character names"],
            "emotional_impact": "string describing emotional impact",
            "tags": ["array of relevant tags"],
            "timing": "early/middle/late"
        }}

        Include memories about:
        - Major plot events
        - Personal interactions
        - Emotional moments
        - Important decisions
        - Character development
        - Significant realizations
        
        FORMAT YOUR RESPONSE AS A VALID JSON ARRAY ONLY, starting with [ and ending with ].
        """

        # Process content in smaller chunks
        chunk_size = 2000  # Smaller chunk size for better processing
        content_chunks = [epub_content[i:i + chunk_size] 
                        for i in range(0, len(epub_content), chunk_size)]
        
        all_memories = []
        for i, chunk in enumerate(content_chunks):
            try:
                chunk_prompt = f"""{base_prompt}

    Text content (Part {i+1}/{len(content_chunks)}):
    {chunk}

    Remember to respond ONLY with a JSON array of memory objects.
    """
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a JSON-generating assistant that creates memory entries. Always respond with valid JSON arrays only."
                        },
                        {"role": "user", "content": chunk_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )

                response_text = response.choices[0].message.content.strip()
                
                # Clean up the response to ensure it's valid JSON
                try:
                    # First try to parse as-is
                    chunk_memories = json.loads(response_text)
                    if not isinstance(chunk_memories, list):
                        chunk_memories = [chunk_memories]
                except json.JSONDecodeError:
                    # If that fails, try to extract JSON array
                    import re
                    json_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
                    if json_match:
                        chunk_memories = json.loads(f"[{json_match.group(1)}]")
                    else:
                        print(f"Chunk {i+1} did not return valid JSON. Skipping...")
                        continue

                all_memories.extend(chunk_memories)
                print(f"Successfully processed chunk {i+1}, found {len(chunk_memories)} memories")
                
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_memories = []
        for memory in all_memories:
            memory_key = memory['description']
            if memory_key not in seen:
                seen.add(memory_key)
                unique_memories.append(memory)
        
        # Sort memories chronologically
        timing_order = {'early': 0, 'middle': 1, 'late': 2}
        unique_memories.sort(key=lambda x: timing_order[x['timing']])
        
        print(f"Total unique memories extracted: {len(unique_memories)}")
        return unique_memories

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

    def process_epub(self, epub_path: str, character_name: str) -> Dict:
        """Process epub file and generate personality output."""
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
        
        # Ensure all new fields are present
        full_personality = {
            "tone": personality.get("tone", ""),
            "response_style": personality.get("response_style", ""),
            "behavior": personality.get("behavior", ""),
            "user_preferences": personality.get("user_preferences", {
                "likes": [],
                "dislikes": [],
                "preferences": {}
            }),
            "do_dont": personality.get("do_dont", {
                "do": [],
                "dont": []
            }),
            "personality_traits": personality.get("personality_traits", {
                "strengths": [],
                "weaknesses": [],
                "growth_areas": [],
                "core_values": [],
                "coping_mechanisms": []
            }),
            "background_influence": personality.get("background_influence", {
                "key_experiences": [],
                "relationships": [],
                "worldview": ""
            }),
            "social_dynamics": personality.get("social_dynamics", {
                "leadership_style": "",
                "group_role": "",
                "friendship_approach": "",
                "trust_patterns": "",
                "conflict_style": ""
            }),
            "name": character_name
        }
        
        return full_personality


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
        personality = extractor.process_epub(epub_path, character_name)
        
        # Save personality.json
        output_file = f"{character_name.lower()}_personality.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(personality, f, indent=4, ensure_ascii=False)
            
        print(f"Successfully created personality file: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise  # This will show the full error trace

if __name__ == "__main__":
    main()