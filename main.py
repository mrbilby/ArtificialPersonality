import json
from typing import List
from openai import OpenAI
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

    @staticmethod
    def load_from_file(file_path="personality.json"):
        with open(file_path, "r") as file:
            data = json.load(file)
        profile = PersonalityProfile()
        profile.__dict__.update(data)
        return profile


class Memory:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.entries = []

    def add_entry(self, entry: str):
        # Simple token count simulation (1 word = 1 token for simplicity)
        tokens = len(entry.split())
        if tokens > self.max_tokens:
            raise ValueError("Entry exceeds maximum token limit.")
        self.entries.append(entry)
        self._enforce_token_limit()

    def retrieve_relevant_entries(self, query: str, top_n=3) -> List[str]:
        # Naive relevance check based on word overlap
        scores = [
            (
                entry,
                sum(
                    1
                    for word in query.lower().split()
                    if word in entry.lower()
                ),
            )
            for entry in self.entries
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, score in scores[:top_n] if score > 0]

    def _enforce_token_limit(self):
        total_tokens = sum(len(entry.split()) for entry in self.entries)
        while total_tokens > self.max_tokens and self.entries:
            removed = self.entries.pop(0)  # Remove oldest entry
            total_tokens = sum(len(entry.split()) for entry in self.entries)


class ShortTermMemory(Memory):
    def __init__(self, max_tokens: int = 1000):
        super().__init__(max_tokens)


class LongTermMemory(Memory):
    def __init__(self, max_tokens: int = 10000):
        super().__init__(max_tokens)


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

    def process_query(self, query: str) -> str:
        # Retrieve relevant context
        short_term_context = self.short_memory.retrieve_relevant_entries(query)
        long_term_context = self.long_memory.retrieve_relevant_entries(query)

        # Combine personality, context, and query into messages
        messages = self._build_messages(query, short_term_context, long_term_context)

        # Generate response using OpenAI's API
        try:
            response = self._generate_response(messages)
            response_text = response.choices[0].message.content.strip()
        except Exception as e:
            response_text = f"Sorry, I encountered an error: {str(e)}"

        # Add to short-term memory
        self.short_memory.add_entry(f"User: {query}")
        self.short_memory.add_entry(f"Bot: {response_text}")

        # Decide whether to move entries from short-term to long-term memory
        self._manage_memory()

        return response_text

    def _generate_response(self, messages: List[dict]) -> object:
        """
        Generate a response using OpenAI's ChatCompletion API.
        """
        response = self.client.chat.completions.create(
            model="gpt-4",  # Use "gpt-4" or the appropriate model name
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            n=1,
            stop=None,
        )
        return response

    def _build_messages(
        self, query: str, short_context: List[str], long_context: List[str]
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
        context = "\n".join(short_context + long_context)
        if context:
            system_content += f"\nContext:\n{context}\n"

        # Combine into messages
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        return messages

    def _manage_memory(self):
        """
        Transfer entries from short-term to long-term memory based on simple criteria.
        For POC purposes, we'll transfer every 5 interactions.
        """
        # Count the number of interactions in short-term memory
        interactions = len(self.short_memory.entries) // 2  # Each interaction has User and Bot entries

        if interactions >= 5:
            # Transfer the last 5 interactions to long-term memory
            entries_to_transfer = self.short_memory.entries[-10:]  # 5 interactions x 2
            for entry in entries_to_transfer:
                self.long_memory.add_entry(entry)
            # Remove them from short-term memory
            self.short_memory.entries = self.short_memory.entries[:-10]
            print("[Memory Management] Transferred last 5 interactions to long-term memory.")

    def transfer_all_memories(self):
        """
        Transfer all short-term memories to long-term memory.
        """
        if self.short_memory.entries:
            for entry in self.short_memory.entries:
                self.long_memory.add_entry(entry)
            self.short_memory.entries = []
            print("[Memory Management] All short-term memories have been transferred to long-term memory.")
        else:
            print("[Memory Management] No short-term memories to transfer.")



def main():
    print("Initializing ChatBot...")

    # Initialize personality
    personality = PersonalityProfile(
        tone="casual",
        response_style="detailed",
        behavior="proactive",
        user_preferences={
            "project_preferences": "I prefer working on AI and machine learning projects."
        },
    )
    # Optionally, add some do/don't rules
    personality.add_do_rule("Provide detailed explanations.")
    personality.add_dont_rule("Use overly technical jargon without explanation.")

    # Initialize memories
    short_memory = ShortTermMemory(max_tokens=1000)
    long_memory = LongTermMemory(max_tokens=10000)

    # Load existing personality and memories if available
    try:
        personality = PersonalityProfile.load_from_file()
        print("[Info] Loaded existing personality profile.")
    except FileNotFoundError:
        print("[Info] No existing personality profile found. Using default settings.")

    try:
        with open("short_memory.json", "r") as f:
            short_memory.entries = json.load(f)
        print("[Info] Loaded existing short-term memory.")
    except FileNotFoundError:
        print("[Info] No existing short-term memory found. Starting fresh.")

    try:
        with open("long_memory.json", "r") as f:
            long_memory.entries = json.load(f)
        print("[Info] Loaded existing long-term memory.")
    except FileNotFoundError:
        print("[Info] No existing long-term memory found. Starting fresh.")

    # Initialize chatbot
    try:
        chatbot = ChatBot(personality, short_memory, long_memory)
    except ValueError as e:
        print(f"[Error] {e}")
        print("Please ensure that the API_KEY is set in the .env file.")
        return

    print(
        "ChatBot is ready! Type your messages below. Type 'exit' or 'bye' to end the conversation.\n"
    )

    while True:
        try:
            user_input = input("You: ").strip()
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
            json.dump(chatbot.short_memory.entries, f, indent=4)
    except Exception as e:
        print(f"Failed to save short-term memory: {e}")

    try:
        with open("long_memory.json", "w") as f:
            json.dump(chatbot.long_memory.entries, f, indent=4)
    except Exception as e:
        print(f"Failed to save long-term memory: {e}")

    print("Session saved. Goodbye!")


if __name__ == "__main__":
    main()
