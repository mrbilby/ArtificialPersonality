import json
from datetime import datetime

def clean_long_term_memory(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    cleaned_interactions = []
    for entry in data:
        interaction = entry.copy()

        # Ensure 'user_message' and 'bot_response' are present
        if 'user_message' not in interaction or not interaction['user_message']:
            print("Missing 'user_message'. Skipping this interaction.")
            continue
        if 'bot_response' not in interaction or not interaction['bot_response']:
            print("Missing 'bot_response'. Skipping this interaction.")
            continue

        # Validate and fix 'timestamp'
        timestamp_str = interaction.get('timestamp')
        if not timestamp_str:
            print("Missing 'timestamp'. Setting to current time.")
            interaction['timestamp'] = datetime.now().isoformat()
        else:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp > datetime.now():
                    print(f"Future timestamp detected: {timestamp_str}. Adjusting to current time.")
                    interaction['timestamp'] = datetime.now().isoformat()
                else:
                    interaction['timestamp'] = timestamp_str  # Keep the valid timestamp
            except ValueError:
                print(f"Invalid timestamp format: {timestamp_str}. Setting to current time.")
                interaction['timestamp'] = datetime.now().isoformat()

        # Validate and fix 'tags'
        tags = interaction.get('tags')
        if not tags or not isinstance(tags, list):
            print("Missing or invalid 'tags'. Generating tags from messages.")
            user_message = interaction.get('user_message', '')
            bot_response = interaction.get('bot_response', '')
            combined_text = user_message + ' ' + bot_response
            # Implement your tag generation logic here
            generated_tags = generate_tags(combined_text)
            interaction['tags'] = generated_tags if generated_tags else ['general']

        # Ensure 'priority_score' and 'priority_factors' are present
        if 'priority_score' not in interaction:
            interaction['priority_score'] = 0.5  # Assign a default value
        if 'priority_factors' not in interaction:
            interaction['priority_factors'] = {}

        cleaned_interactions.append(interaction)

    # Write cleaned data back to file
    with open(file_path, 'w') as f:
        json.dump(cleaned_interactions, f, indent=4)
    print("Long-term memory cleaned successfully.")

def generate_tags(text):
    # Implement your tag generation logic here
    # For demonstration, we'll return a placeholder list
    return ['general']

# Use the function
clean_long_term_memory('carl_long_term_memory.json')
