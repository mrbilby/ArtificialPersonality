import json
from typing import List, Dict
from openai import OpenAI  # As per your working setup
from dotenv import load_dotenv
import os

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
    
    def to_dict(self):
        return {
            "user_message": self.user_message,
            "bot_response": self.bot_response,
            "tags": self.tags
        }
    
    @staticmethod
    def from_dict(data: Dict):
        return Interaction(
            user_message=data["user_message"],
            bot_response=data["bot_response"],
            tags=data.get("tags", [])
        )

class Memory:
    def __init__(self, max_interactions: int):
        self.max_interactions = max_interactions
        self.interactions: List[Interaction] = []
    
    def add_interaction(self, interaction: Interaction):
        self.interactions.append(interaction)
        print(f"[Debug] Interaction added: {interaction.user_message}")
        self._enforce_limit()
    
    def retrieve_relevant_interactions(self, query: str, top_n=3) -> List[Interaction]:
        # Simple keyword-based relevance
        scores = [
            (interaction, sum(word.lower() in interaction.user_message.lower() for word in query.split()))
            for interaction in self.interactions
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        relevant = [interaction for interaction, score in scores[:top_n] if score > 0]
        print(f"[Debug] Retrieved {len(relevant)} relevant interactions.")
        return relevant
    
    def _enforce_limit(self):
        while len(self.interactions) > self.max_interactions:
            removed = self.interactions.pop(0)  # Remove oldest interaction
            print(f"[Debug] Removed from memory: {removed.user_message}")
    
    def to_list(self):
        return [interaction.to_dict() for interaction in self.interactions]
    
    def load_from_list(self, data: List[Dict]):
        self.interactions = [Interaction.from_dict(entry) for entry in data]
        print(f"[Debug] Loaded {len(self.interactions)} interactions into memory.")


class ShortTermMemory(Memory):
    def __init__(self, max_interactions: int = 10):
        super().__init__(max_interactions)


class LongTermMemory(Memory):
    def __init__(self, max_interactions: int = 1000):
        super().__init__(max_interactions)
    
    def retrieve_relevant_interactions_by_tags(self, tags: List[str], top_n=5) -> List[Interaction]:
        relevant = []
        for interaction in self.interactions:
            if any(tag in interaction.tags for tag in tags):
                relevant.append(interaction)
            if len(relevant) >= top_n:
                break
        print(f"[Debug] Retrieved {len(relevant)} interactions from Long-Term Memory based on tags.")
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
        if not self.api_key:
            raise ValueError("API_KEY not found in environment variables.")
        self.client = OpenAI(api_key=self.api_key)
        print("[Debug] OpenAI client initialized.")
    
    def process_query(self, query: str) -> str:
        # Retrieve relevant context based on tags
        relevant_tags = self._extract_relevant_tags(query)
        short_term_context = self.short_memory.retrieve_relevant_interactions(query)
        long_term_context = self.long_memory.retrieve_relevant_interactions_by_tags(relevant_tags)
    
        # Combine personality, context, and query into messages
        messages = self._build_messages(query, short_term_context, long_term_context)
    
        # Generate response using OpenAI's API
        try:
            response = self._generate_response(messages)
            response_text = response.choices[0].message.content.strip()
            print("[Debug] Response generated.")
        except Exception as e:
            response_text = f"Sorry, I encountered an error: {str(e)}"
            print(f"[Error] {response_text}")
    
        # Generate tags for the interaction
        tags = self._generate_tags(query)
    
        # Create interaction with tags
        interaction = Interaction(user_message=query, bot_response=response_text, tags=tags)
    
        # Add to long-term memory
        self.long_memory.add_interaction(interaction)
        print(f"[Debug] Added to Long-Term Memory: {interaction.user_message} with tags {interaction.tags}")
    
        # Optionally, add to short-term memory as well
        self.short_memory.add_interaction(interaction)
        print(f"[Debug] Added to Short-Term Memory: {interaction.user_message}")
    
        return response_text
    
    def _build_messages(
        self, query: str, short_context: List[Interaction], long_context: List[Interaction]
    ) -> List[dict]:
        # Construct the system message with personality and rules
        system_content = (
            f"Tone: {self.personality.tone}\n"
            f"Response Style: {self.personality.response_style}\n"
            f"Behavior: {self.personality.behavior}\n\n"
        )
    
        # Add do rules
        if self.personality.do_dont["do"]:
            do_rules = "\n".join([f"Do: {rule}" for rule in self.personality.do_dont["do"]])
            system_content += f"{do_rules}\n"
    
        # Add don't rules
        if self.personality.do_dont["dont"]:
            dont_rules = "\n".join([f"Don't: {rule}" for rule in self.personality.do_dont["dont"]])
            system_content += f"{dont_rules}\n"
    
        # Add user preferences
        if self.personality.user_preferences:
            user_prefs = "\n".join(
                [f"{key}: {value}" for key, value in self.personality.user_preferences.items()]
            )
            system_content += f"{user_prefs}\n"
    
        # Add context
        context = "\n".join(
            [f"User: {inter.user_message}\nBot: {inter.bot_response}" for inter in short_context + long_context]
        )
        if context:
            system_content += f"\nContext:\n{context}\n"
    
        # Combine into messages
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
        """
        Extract relevant tags/topics from the current query using OpenAI's API.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Identify the main topics from the following text."},
                    {"role": "user", "content": query}
                ],
                max_tokens=60,
                temperature=0.3,
            )
            tags_text = response.choices[0].message.content.strip()
            tags = [tag.strip().lower() for tag in tags_text.split(',')]
            print(f"[Debug] Extracted relevant tags for recall: {tags}")
            return tags
        except Exception as e:
            print(f"[Error] Failed to extract relevant tags: {e}")
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
            if user_input.lower() in ["exit", "bye"]:
                response = chatbot.process_query(user_input)
                print(f"Bot: {response}")
                
                # No need to transfer memories manually as all are saved
                break

            if not user_input:
                print("Bot: Please enter a message.")
                continue

            response = chatbot.process_query(user_input)
            print(f"Bot: {response}")

        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye! Have a great day!")
            # No need to transfer memories manually as all are saved
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
