import unittest
from datetime import datetime
from main import ChatBot, PersonalityProfile, ShortTermMemory, LongTermMemory, Interaction, MemoryPriority

class EmotionAwarenessTests(unittest.TestCase):
    def setUp(self):
        # Create a test personality
        self.personality = PersonalityProfile(
            tone="empathetic",
            response_style="emotionally aware",
            behavior="responsive",
            name="test_personality"
        )
        self.short_memory = ShortTermMemory(max_interactions=25)
        self.long_memory = LongTermMemory(max_interactions=1000, personality_name="test_personality")
        self.chatbot = ChatBot(self.personality, self.short_memory, self.long_memory)
        self.priority_system = MemoryPriority()

    def test_basic_emotion_detection(self):
        """Test detection of basic emotions in user messages"""
        test_cases = [
            ("I am so happy today!", "joy"),
            ("This makes me really sad.", "sadness"),
            ("I'm furious about this!", "anger"),
            ("Wow, I can't believe it!", "surprise"),
            ("The weather is cloudy.", "neutral")
        ]

        for message, expected_emotion in test_cases:
            detected_emotion = self.priority_system._detect_emotion(message)
            self.assertEqual(
                detected_emotion, 
                expected_emotion, 
                f"Failed to detect {expected_emotion} in message: {message}"
            )

    def test_emotion_priority_scoring(self):
        """Test if emotional content affects memory priority"""
        emotional_message = Interaction(
            user_message="I'm absolutely thrilled about this achievement!",
            bot_response="That's wonderful news!",
            tags=["joy", "achievement"]
        )
        neutral_message = Interaction(
            user_message="The meeting is scheduled for tomorrow.",
            bot_response="Noted, I'll remember that.",
            tags=["schedule", "meeting"]
        )

        # Calculate priority scores
        emotional_score, _ = self.priority_system.calculate_priority_with_factors(
            emotional_message, {"recent_interactions": []}
        )
        neutral_score, _ = self.priority_system.calculate_priority_with_factors(
            neutral_message, {"recent_interactions": []}
        )

        self.assertGreater(
            emotional_score, 
            neutral_score, 
            "Emotional messages should have higher priority scores"
        )

    def test_emotional_context_preservation(self):
        """Test if emotional context is preserved in memory"""
        # Add a sequence of emotional interactions
        emotional_sequence = [
            ("I'm so excited about this project!", "joy"),
            ("Everything went wrong today.", "sadness"),
            ("This is absolutely incredible!", "surprise")
        ]

        for message, emotion in emotional_sequence:
            interaction = Interaction(
                user_message=message,
                bot_response="I understand how you feel.",
                tags=[emotion]
            )
            self.long_memory.add_interaction(interaction)

        # Retrieve memories with matching emotional context
        relevant = self.long_memory.retrieve_relevant_interactions_by_tags(["joy", "sadness"])
        
        self.assertGreaterEqual(
            len(relevant), 
            2, 
            "Should retrieve memories with matching emotional context"
        )

    def test_emotional_response_consistency(self):
        """Test if chatbot maintains emotional consistency in responses"""
        # Simulate a conversation with emotional context
        conversation = [
            "I'm feeling really down today.",
            "My dog just passed away.",
            "I don't know how to handle this."
        ]

        responses = []
        for message in conversation:
            response = self.chatbot.process_query(message)
            responses.append(response)

        # Check for emotional awareness in responses
        empathy_keywords = ['understand', 'sorry', 'hear', 'difficult', 'support']
        for response in responses:
            has_empathy = any(keyword in response.lower() for keyword in empathy_keywords)
            self.assertTrue(
                has_empathy, 
                f"Response lacks emotional awareness: {response}"
            )

    def test_emotion_intensity_scaling(self):
        """Test if emotion intensity is properly scaled"""
        test_cases = [
            ("I'm slightly annoyed.", "anger", 0.5),
            ("I'm absolutely furious!", "anger", 0.9),
            ("This is the happiest day of my life!", "joy", 0.9),
            ("I'm a bit happy.", "joy", 0.5)
        ]

        for message, emotion, expected_intensity in test_cases:
            interaction = Interaction(
                user_message=message,
                bot_response="I understand.",
                tags=[emotion]
            )
            score, factors = self.priority_system.calculate_priority_with_factors(
                interaction, {"recent_interactions": []}
            )
            self.assertAlmostEqual(
                factors['emotional_impact'],
                expected_intensity,
                delta=0.4,
                msg=f"Emotion intensity not properly scaled for: {message}"
            )

if __name__ == '__main__':
    unittest.main(verbosity=2)