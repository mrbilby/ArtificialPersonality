import json
from openai import OpenAI  # As per your working setup
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import statistics
from typing import List, Dict, Optional

# Load environment variables from .env file
load_dotenv()


class PersonalityProfile:
    def __init__(
        self,
        tone="neutral",
        response_style="detailed",
        behavior="reactive",
        user_preferences=None,
    ):
        self.tone = tone
        self.response_style = response_style
        self.behavior = behavior
        self.user_preferences = user_preferences or {}
        self.do_dont = {"do": [], "dont": []}

    def update_tone(self, new_tone):
        self.tone = new_tone

    def update_response_style(self, new_style):
        self.response_style = new_style

    def add_user_preference(self, key, value):
        self.user_preferences[key] = value

    def add_do_rule(self, rule: str):
        self.do_dont["do"].append(rule)

    def add_dont_rule(self, rule: str):
        self.do_dont["dont"].append(rule)

    def save_to_file(self, file_path="personality.json"):
        with open(file_path, "w") as file:
            json.dump(self.__dict__, file, indent=4)
        print("[Debug] Personality profile saved.")

    @staticmethod
    def load_from_file(file_path="personality.json"):
        with open(file_path, "r") as file:
            data = json.load(file)
        profile = PersonalityProfile()
        profile.__dict__.update(data)
        print("[Debug] Personality profile loaded.")
        return profile


class Interaction:
    def __init__(self, user_message: str, bot_response: str, tags: List[str] = None):
        self.user_message = user_message
        self.bot_response = bot_response
        self.tags = tags or []
        # Store timestamps as datetime objects immediately
        self.timestamp = datetime.now()
    
    def to_dict(self):
        return {
            "user_message": self.user_message,
            "bot_response": self.bot_response,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat()  # Convert to ISO format for storage
        }
    
    @staticmethod
    def from_dict(data: Dict):
        interaction = Interaction(
            user_message=data["user_message"],
            bot_response=data["bot_response"],
            tags=data.get("tags", [])
        )
        # Convert ISO format back to datetime
        interaction.timestamp = datetime.fromisoformat(data["timestamp"])
        return interaction
    
class TimeAnalyzer:
    @staticmethod
    def analyze_patterns(interactions: List[Interaction]) -> Dict:
        patterns = {
            'current_response_time': None,
            'average_response_time': None,
            'last_message_age': None,
            'time_since_last_chat': None,
            'interaction_frequency': 'first_time',
            'familiarity_level': 'new',
            'total_interactions': 0,
            'typical_chat_time': None
        }

        if not interactions or len(interactions) < 2:
            return patterns

        # Sort interactions by timestamp
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        
        # Get current time
        now = datetime.now()
        
        try:
            # Time since the most recent message
            last_message = sorted_interactions[-1]
            patterns['last_message_age'] = (now - last_message.timestamp).total_seconds()
            
            # Time between the two most recent messages
            last_two_messages = sorted_interactions[-2:]
            if len(last_two_messages) == 2:
                time_between = (last_two_messages[1].timestamp - 
                              last_two_messages[0].timestamp).total_seconds()
                patterns['current_response_time'] = time_between
            
            # Calculate recent response times (last 5 pairs)
            recent_times = []
            for i in range(len(sorted_interactions)-1):
                time_diff = (sorted_interactions[i+1].timestamp - 
                           sorted_interactions[i].timestamp).total_seconds()
                recent_times.append(time_diff)
            
            if recent_times:
                # Get last 5 response times
                last_five = recent_times[-5:]
                patterns['average_response_time'] = statistics.mean(last_five)

            # Debug output
            print(f"[Debug] Time Analysis:")
            print(f"[Debug] Last message timestamp: {last_message.timestamp}")
            print(f"[Debug] Current time: {now}")
            print(f"[Debug] Time since last message: {patterns['last_message_age']:.2f} seconds")
            if patterns['current_response_time']:
                print(f"[Debug] Time between last two messages: {patterns['current_response_time']:.2f} seconds")
            if patterns['average_response_time']:
                print(f"[Debug] Average response time: {patterns['average_response_time']:.2f} seconds")

        except Exception as e:
            print(f"[Debug] Error in time analysis: {e}")
            
        patterns['total_interactions'] = len(sorted_interactions)

        return patterns


class Memory:
    def __init__(self, max_interactions: int):
        self.max_interactions = max_interactions
        self.interactions: List[Interaction] = []
    
    def add_interaction(self, interaction: Interaction):
        self.interactions.append(interaction)
        print(f"[Debug] Interaction added: {interaction.user_message}")
        self._enforce_limit()
    
    def retrieve_relevant_interactions(self, query: str, top_n=10) -> List[Interaction]:
        """
        Returns the last 5 interactions and any additional interactions with matching words
        """
        # Get the last 5 interactions
        recent = self.interactions[-5:] if len(self.interactions) >= 5 else self.interactions[:]
        
        # Score remaining interactions
        older_interactions = self.interactions[:-5] if len(self.interactions) > 5 else []
        scores = [
            (interaction, sum(word.lower() in interaction.user_message.lower() for word in query.split()))
            for interaction in older_interactions
        ]
        
        # Get any older interactions with matches
        relevant = [interaction for interaction, score in scores if score > 0]
        
        # Combine recent and relevant (avoiding duplicates while preserving order)
        combined = []
        seen = set()
        for interaction in recent + relevant:
            if interaction not in seen:
                combined.append(interaction)
                seen.add(interaction)
        
        print(f"[Debug] Retrieved {len(recent)} recent and {len(relevant)} relevant older interactions.")
        return combined[:top_n]
    
    def _enforce_limit(self):
        while len(self.interactions) > self.max_interactions:
            removed = self.interactions.pop(0)  # Remove oldest interaction
            print(f"[Debug] Removed from memory: {removed.user_message}")
    
    def to_list(self):
        return [interaction.to_dict() for interaction in self.interactions]
    
    def load_from_list(self, data: List[Dict]):
        # Add debug printing to check loaded timestamps
        print("[Debug] Loading memory with timestamps:")
        self.interactions = []
        for entry in data:
            try:
                interaction = Interaction.from_dict(entry)
                print(f"[Debug] Loaded interaction timestamp: {interaction.timestamp}")
                self.interactions.append(interaction)
            except Exception as e:
                print(f"[Debug] Error loading interaction: {e}")
        print(f"[Debug] Loaded {len(self.interactions)} interactions into memory.")


class ShortTermMemory(Memory):
    def __init__(self, max_interactions: int = 10):
        super().__init__(max_interactions)


class LongTermMemory(Memory):
    def __init__(self, max_interactions: int = 1000):
        super().__init__(max_interactions)
    
    def retrieve_relevant_interactions_by_tags(self, tags: List[str], top_n=5) -> List[Interaction]:
        # Add fuzzy matching for tags
        relevant_scores = []
        for interaction in self.interactions:
            score = 0
            for tag in tags:
                # Check for partial matches
                for interaction_tag in interaction.tags:
                    if (tag in interaction_tag or interaction_tag in tag):
                        score += 1
            if score > 0:
                relevant_scores.append((interaction, score))
        
        # Sort by relevance score
        relevant_scores.sort(key=lambda x: x[1], reverse=True)
        relevant = [interaction for interaction, score in relevant_scores[:top_n]]
        print(f"[Debug] Retrieved {len(relevant)} interactions from Long-Term Memory with scores: {[score for _, score in relevant_scores[:top_n]]}")
        return relevant


class ChatBot:
    def __init__(
        self,
        personality: PersonalityProfile,
        short_memory: ShortTermMemory,
        long_memory: LongTermMemory,
    ):
        self.personality = personality
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.api_key = os.getenv("API_KEY")
        self.time_analyzer = TimeAnalyzer() 
        if not self.api_key:
            raise ValueError("API_KEY not found in environment variables.")
        self.client = OpenAI(api_key=self.api_key)
        print("[Debug] OpenAI client initialized.")

    def print_memory_status(self):
        """Add this method to the ChatBot class"""
        print("\n=== Memory Status ===")
        print("Short-term memory:")
        for i, interaction in enumerate(self.short_memory.interactions[-5:], 1):
            print(f"{i}. Time: {interaction.timestamp}, Message: {interaction.user_message[:30]}...")
        
        print("\nLong-term memory:")
        for i, interaction in enumerate(sorted(self.long_memory.interactions[-5:], 
                                            key=lambda x: x.timestamp), 1):
            print(f"{i}. Time: {interaction.timestamp}, Message: {interaction.user_message[:30]}...")
        print("==================\n")

    def process_query(self, query: str) -> str:
        # Add timestamp debugging
        print(f"[Debug] Processing query at: {datetime.now()}")
        
        # Create interaction first to capture accurate timestamp
        interaction = Interaction(user_message=query, bot_response="", tags=[])
        current_time = interaction.timestamp
        
        # Get relevant context
        relevant_tags = self._extract_relevant_tags(query)
        short_term_context = self.short_memory.retrieve_relevant_interactions(query)
        long_term_context = self.long_memory.retrieve_relevant_interactions_by_tags(relevant_tags)
        
        # Add the new interaction to memories before generating response
        self.short_memory.add_interaction(interaction)
        self.long_memory.add_interaction(interaction)
        
        # Generate response
        messages = self._build_messages(query, short_term_context, long_term_context)
        try:
            response = self._generate_response(messages)
            response_text = response.choices[0].message.content.strip()
            print("[Debug] Response generated.")
        except Exception as e:
            response_text = f"Sorry, I encountered an error: {str(e)}"
            print(f"[Error] {response_text}")
        
        # Update the interaction with the response
        interaction.bot_response = response_text
        
        # Generate and update tags
        tags = self._generate_tags(query)
        interaction.tags = tags
        
        print(f"[Debug] Interaction completed at: {datetime.now()}")
        return response_text
        
    def _format_time(self, seconds: float) -> str:
        """Format seconds into a readable string with better precision"""
        if seconds < 1:
            return "less than a second"
        elif seconds < 60:
            return f"{seconds:.1f} seconds"  # Keep one decimal place for seconds
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            if remaining_seconds == 0:
                return f"{minutes} minute{'s' if minutes != 1 else ''}"
            return f"{minutes} minute{'s' if minutes != 1 else ''} and {remaining_seconds} seconds"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            return f"{hours} hour{'s' if hours != 1 else ''} and {remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}"


    def _build_messages(self, query: str, short_context: List[Interaction], 
                    long_context: List[Interaction]) -> List[dict]:
        patterns = self.time_analyzer.analyze_patterns(self.long_memory.interactions)
        print(f"[Debug] Including {len(short_context)} short-term and {len(long_context)} long-term memories in context")

        system_content = (
            f"Tone: {self.personality.tone}\n"
            f"Response Style: {self.personality.response_style}\n"
            f"Behavior: {self.personality.behavior}\n\n"
            "Time Context (BE PRECISE WITH THESE TIMES):\n"  # Added emphasis on precision
        )
        
        if patterns.get('current_response_time') is not None:
            system_content += (
                f"- Exact time between the previous two messages: "
                f"{self._format_time(patterns['current_response_time'])}\n"
            )
        
        if patterns.get('last_message_age') is not None:
            system_content += (
                f"- Exact time since the most recent message: "
                f"{self._format_time(patterns['last_message_age'])}\n"
            )
        
        if patterns.get('average_response_time') is not None:
            system_content += (
                f"- Average response time (last 5 messages): "
                f"{self._format_time(patterns['average_response_time'])}\n"
            )
        
        # Add other context information
        system_content += f"- Interaction frequency: {patterns.get('interaction_frequency', 'first_time')}\n"
        system_content += f"- Familiarity level: {patterns.get('familiarity_level', 'new')}\n"
        system_content += f"- Total interactions: {patterns.get('total_interactions', 0)}\n"
        
        if patterns.get('typical_chat_time') is not None:
            system_content += f"- Chatting at typical time: {patterns['typical_chat_time']}\n"
        
        # Add do/don't rules
        if self.personality.do_dont["do"]:
            do_rules = "\n".join([f"Do: {rule}" for rule in self.personality.do_dont["do"]])
            system_content += f"\n{do_rules}\n"
        
        if self.personality.do_dont["dont"]:
            dont_rules = "\n".join([f"Don't: {rule}" for rule in self.personality.do_dont["dont"]])
            system_content += f"\n{dont_rules}\n"
        
        # Add user preferences
        if self.personality.user_preferences:
            user_prefs = "\n".join(
                [f"{key}: {value}" for key, value in self.personality.user_preferences.items()]
            )
            system_content += f"\n{user_prefs}\n"
        
        # Add context with timestamps
        context = "\n".join(
            [f"[{inter.timestamp}]\nUser: {inter.user_message}\nBot: {inter.bot_response}" 
            for inter in short_context + long_context]
        )
        
        if context:
            system_content += f"\nContext:\n{context}\n"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        return messages
    
    def _generate_response(self, messages: List[dict]) -> dict:
        """
        Generate a response using OpenAI's ChatCompletion API.
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Use your desired model
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            n=1,
            stop=None,
        )
        return response
    

    def _generate_tags(self, user_message: str) -> List[str]:
        """
        Generate clean, relevant tags for a user message using OpenAI's API.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract relevant topics from the following text as a comma-separated list without additional descriptions or numbering."},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=60,
                temperature=0.3,
            )
            tags_text = response.choices[0].message.content.strip()
            # Split by comma and clean each tag
            tags = [tag.strip().lower() for tag in tags_text.split(',') if tag.strip()]
            print(f"[Debug] Generated tags for message: {tags}")
            return tags
        except Exception as e:
            print(f"[Error] Failed to generate tags: {e}")
            return []


    
    def _extract_relevant_tags(self, query: str) -> List[str]:
        """Modified tag extraction to include broader categories"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """
                        Extract topics from the text at multiple levels:
                        1. Specific topics (e.g., 'pineapple pizza')
                        2. General categories (e.g., 'food preferences')
                        3. Emotional context (e.g., 'likes', 'dislikes')
                        Return as comma-separated list.
                    """},
                    {"role": "user", "content": query}
                ],
                max_tokens=60,
                temperature=0.3,
            )
            tags = [tag.strip().lower() for tag in response.choices[0].message.content.split(',')]
            print(f"[Debug] Extracted hierarchical tags: {tags}")
            return tags
        except Exception as e:
            print(f"[Error] Failed to extract tags: {e}")
            return []
    
    def _manage_memory(self):
        """
        Not needed anymore as we're saving all interactions with tags.
        """
        pass  # No action needed since all interactions are stored with tags
    
    def transfer_all_memories(self):
        """
        Optionally, you can implement additional memory transfer logic if needed.
        """
        pass  # Not required in this new system

def main():
    print("Initializing ChatBot...")

    # Initialize personality
    personality = PersonalityProfile(
        tone="casual",
        response_style="detailed",
        behavior="proactive",
        user_preferences={"project_preferences": "I prefer working on AI and machine learning projects."},
    )
    # Add do/don't rules
    personality.add_do_rule("Provide detailed explanations.")
    personality.add_dont_rule("Use overly technical jargon without explanation.")

    # Initialize memories
    short_memory = ShortTermMemory(max_interactions=10)
    long_memory = LongTermMemory(max_interactions=100)

    # Load existing personality and memories if available
    try:
        personality = PersonalityProfile.load_from_file()
    except FileNotFoundError:
        print("[Info] No existing personality profile found. Using default settings.")

    try:
        with open("short_memory.json", "r") as f:
            short_memory.load_from_list(json.load(f))
    except FileNotFoundError:
        print("[Info] No existing short-term memory found. Starting fresh.")

    try:
        with open("long_memory.json", "r") as f:
            long_memory.load_from_list(json.load(f))
    except FileNotFoundError:
        print("[Info] No existing long-term memory found. Starting fresh.")

    # Initialize chatbot
    try:
        chatbot = ChatBot(personality, short_memory, long_memory)
    except ValueError as e:
        print(f"[Error] {e}")
        print("Please ensure that the API_KEY is set in the .env file.")
        return

    print("ChatBot is ready! Type your messages below. Type 'exit' or 'bye' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            
            # Add this new condition before the exit check
            if user_input.lower() == "debug":
                chatbot.print_memory_status()
                continue
                
            if user_input.lower() in ["exit", "bye"]:
                response = chatbot.process_query(user_input)
                print(f"Bot: {response}")
                break

            if not user_input:
                print("Bot: Please enter a message.")
                continue

            response = chatbot.process_query(user_input)
            print(f"Bot: {response}")

        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye! Have a great day!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

    # Save personality and memories before exiting
    try:
        personality.save_to_file()
    except Exception as e:
        print(f"Failed to save personality profile: {e}")

    try:
        with open("short_memory.json", "w") as f:
            json.dump(short_memory.to_list(), f, indent=4)
        print("[Debug] Short-term memory saved.")
    except Exception as e:
        print(f"Failed to save short-term memory: {e}")

    try:
        with open("long_memory.json", "w") as f:
            json.dump(long_memory.to_list(), f, indent=4)
        print("[Debug] Long-term memory saved.")
    except Exception as e:
        print(f"Failed to save long-term memory: {e}")

    print("Session saved. Goodbye!")



if __name__ == "__main__":
    main()
