import json
from openai import OpenAI  # As per your working setup
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import statistics
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()


class PersonalityProfile:
    def __init__(
        self,
        tone="neutral",
        response_style="detailed",
        behavior="reactive",
        user_preferences=None,
        name=None,
    ):
        self.tone = tone
        self.response_style = response_style
        self.behavior = behavior
        self.user_preferences = user_preferences or {}
        self.do_dont = {"do": [], "dont": []}
        self.name = name  # Add name attribute for identification

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

    def set_name(self, name: str):
        """Set the personality name"""
        self.name = name

    def get_file_path(self, name: str = None) -> str:
        """Get the appropriate file path for this personality"""
        personality_name = name or self.name or "default"
        return f"{personality_name}_personality.json"

    def save_to_file(self, file_path=None):
        """Save personality to file using either provided path or generated name-based path"""
        save_path = file_path or self.get_file_path()
        data = {
            "tone": self.tone,
            "response_style": self.response_style,
            "behavior": self.behavior,
            "user_preferences": self.user_preferences,
            "do_dont": self.do_dont,
            "name": self.name
        }
        with open(save_path, "w") as file:
            json.dump(data, file, indent=4)
        print(f"[Debug] Personality profile saved to {save_path}")

    @staticmethod
    def load_from_file(file_path: str):
        """Load personality from file with better error handling"""
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            
            profile = PersonalityProfile(
                tone=data.get("tone", "neutral"),
                response_style=data.get("response_style", "detailed"),
                behavior=data.get("behavior", "reactive"),
                user_preferences=data.get("user_preferences", {})
            )
            
            # Load do/don't rules
            profile.do_dont = data.get("do_dont", {"do": [], "dont": []})
            
            # Set name if it exists in the file
            profile.name = data.get("name")
            
            print(f"[Debug] Personality profile loaded from {file_path}")
            return profile
            
        except FileNotFoundError:
            print(f"[Debug] No existing personality file found at {file_path}")
            raise
        except json.JSONDecodeError:
            print(f"[Error] Invalid JSON in personality file: {file_path}")
            raise
        except Exception as e:
            print(f"[Error] Failed to load personality file: {e}")
            raise


class Interaction:
    def __init__(self, user_message: str, bot_response: str, tags: List[str] = None):
        self.user_message = user_message
        self.bot_response = bot_response
        self.tags = tags or []
        self.timestamp = datetime.now()
        self.priority_score = 0.0
        self.priority_factors = {}  # NEW: Track what contributed to priority
    
    def to_dict(self):
        return {
            "user_message": self.user_message,
            "bot_response": self.bot_response,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
            "priority_score": self.priority_score,
            "priority_factors": self.priority_factors  # NEW: Save factors
        }
    
    @staticmethod
    def from_dict(data: Dict):
        interaction = Interaction(
            user_message=data["user_message"],
            bot_response=data["bot_response"],
            tags=data.get("tags", [])
        )
        interaction.timestamp = datetime.fromisoformat(data["timestamp"])
        interaction.priority_score = float(data.get("priority_score", 0.0))  # Ensure float conversion
        interaction.priority_factors = data.get("priority_factors", {})  # NEW: Load factors
        return interaction
    

class MemoryPriority:
    def __init__(self):
        # Define weights for different priority factors
        self.priority_weights = {
            'emotional_impact': 0.25,    # Weight for emotional significance
            'recency': 0.20,            # Weight for how recent the memory is
            'repetition': 0.15,         # Weight for how often topic is discussed
            'contextual_links': 0.20,   # Weight for connections to other memories
            'user_importance': 0.20     # Weight for explicit user emphasis
        }
        
        # Define emotional intensity values
        self.emotion_intensity = {
            'joy': 0.8,
            'sadness': 0.7,
            'anger': 0.9,
            'surprise': 0.6,
            'neutral': 0.3
        }
        
        # Keywords that indicate user emphasis on importance
        self.importance_indicators = {
            'remember': 1.2,
            'important': 1.5,
            'crucial': 1.5,
            'essential': 1.4,
            'key': 1.3,
            'significant': 1.4,
            'vital': 1.5,
            'critical': 1.5,
            'never forget': 1.6
        }

    def calculate_priority_with_factors(self, interaction: 'Interaction', 
                                    memory_context: Dict) -> Tuple[float, Dict]:
        factors = {}
        priority_score = 0.0
        
        # Emotional Impact (increased impact)
        emotion = self._detect_emotion(interaction.user_message)
        emotional_score = self.emotion_intensity.get(emotion, 0.3)
        factors['emotional_impact'] = emotional_score
        priority_score += emotional_score * 0.35  # Increased from 0.25
        
        # Recency (slightly reduced)
        time_diff = (datetime.now() - interaction.timestamp).total_seconds()
        recency_score = 1 / (1 + (time_diff / (24 * 3600)))
        factors['recency'] = recency_score
        priority_score += recency_score * 0.15  # Reduced from 0.20
        
        # User Importance (increased impact)
        importance_score = self._calculate_user_importance(interaction)
        factors['user_importance'] = importance_score
        priority_score += importance_score * 0.30  # Increased from 0.20
        
        # Rest remains the same...
        
        return min(priority_score, 1.0), factors

    def calculate_priority(self, interaction: 'Interaction', memory_context: Dict) -> float:
        """Calculate priority score for a memory"""
        priority_score = 0.0
        
        # Emotional Impact
        emotion = self._detect_emotion(interaction.user_message)
        emotional_score = self.emotion_intensity.get(emotion, 0.3)
        priority_score += emotional_score * self.priority_weights['emotional_impact']
        
        # Recency (normalized between 0-1)
        time_diff = (datetime.now() - interaction.timestamp).total_seconds()
        recency_score = 1 / (1 + (time_diff / (24 * 3600)))  # Decay over 24 hours
        priority_score += recency_score * self.priority_weights['recency']
        
        # Repetition
        repetition_score = self._calculate_repetition(interaction, memory_context)
        priority_score += repetition_score * self.priority_weights['repetition']
        
        # Contextual Links
        context_score = self._calculate_context_links(interaction, memory_context)
        priority_score += context_score * self.priority_weights['contextual_links']
        
        # User Importance
        importance_score = self._calculate_user_importance(interaction)
        priority_score += importance_score * self.priority_weights['user_importance']
        
        return min(priority_score, 1.0)  # Normalize to 0-1 range

    def _calculate_repetition(self, interaction: 'Interaction', memory_context: Dict) -> float:
        """Calculate how often similar topics appear"""
        topic_count = 0
        total_memories = len(memory_context.get('recent_interactions', []))
        
        if total_memories == 0:
            return 0.0
            
        for memory in memory_context.get('recent_interactions', []):
            if any(tag in memory.tags for tag in interaction.tags):
                topic_count += 1
                
        return topic_count / total_memories

    def _calculate_context_links(self, interaction: 'Interaction', memory_context: Dict) -> float:
        """Calculate how well memory connects to other memories"""
        connection_score = 0.0
        key_memories = memory_context.get('key_memories', [])
        
        for memory in key_memories:
            # Check tag overlap
            common_tags = set(interaction.tags) & set(memory.get('tags', []))
            if common_tags:
                connection_score += 0.2 * len(common_tags)
            
            # Check temporal proximity
            time_diff = abs((interaction.timestamp - datetime.fromisoformat(memory['timestamp'])).total_seconds())
            if time_diff < 3600:  # Within an hour
                connection_score += 0.3
            
            # Check conversational flow
            if self._are_messages_related(interaction.user_message, memory.get('message', '')):
                connection_score += 0.5
                
        return min(connection_score, 1.0)

    def _calculate_user_importance(self, interaction: 'Interaction') -> float:
        """Calculate importance based on user's explicit indicators"""
        importance_score = 0.0
        message_lower = interaction.user_message.lower()
        
        for indicator, weight in self.importance_indicators.items():
            if indicator in message_lower:
                importance_score += weight
                
        return min(importance_score, 1.0)

    def _are_messages_related(self, message1: str, message2: str) -> bool:
        """Check if two messages are semantically related"""
        words1 = set(message1.lower().split())
        words2 = set(message2.lower().split())
        overlap = len(words1 & words2) / len(words1 | words2)
        return overlap > 0.2

    def _detect_emotion(self, text: str) -> str:
        """Detect dominant emotion in text"""
        emotion_keywords = {
            'joy': {'happy', 'great', 'excellent', 'wonderful', 'love', 'enjoy'},
            'sadness': {'sad', 'sorry', 'disappointed', 'upset', 'unhappy'},
            'anger': {'angry', 'frustrated', 'annoyed', 'mad', 'hate'},
            'surprise': {'wow', 'amazing', 'unexpected', 'surprised'}
        }
        
        text_lower = text.lower()
        emotion_scores = {
            emotion: sum(1 for word in keywords if word in text_lower)
            for emotion, keywords in emotion_keywords.items()
        }
        
        return max(emotion_scores.items(), key=lambda x: x[1])[0] if any(emotion_scores.values()) else 'neutral'

    def adjust_weights(self, usage_patterns: Dict):
        """Dynamically adjust priority weights based on interaction patterns"""
        if usage_patterns.get('emotional_engagement', 0) > 0.7:
            self.priority_weights['emotional_impact'] = 0.3
            
        if usage_patterns.get('reference_frequency', 0) > 0.5:
            self.priority_weights['contextual_links'] = 0.25
            
        if usage_patterns.get('topic_persistence', 0) > 0.6:
            self.priority_weights['repetition'] = 0.2

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
        self.priority_system = MemoryPriority()
        self.last_priority_update = datetime.now()  # NEW: Track last update
    
    def add_interaction(self, interaction: Interaction):
        try:
            memory_context = self._get_memory_context()
            self._update_priorities()  # Update existing priorities
            
            # Calculate new interaction priority
            priority_score, factors = self.priority_system.calculate_priority_with_factors(
                interaction, memory_context
            )
            interaction.priority_score = priority_score
            interaction.priority_factors = factors
            
            self.interactions.append(interaction)
            print(f"[Debug] Interaction added with priority {interaction.priority_score:.2f}")
            print(f"[Debug] Priority factors: {interaction.priority_factors}")
            
            self._enforce_limit()
        except Exception as e:
            print(f"[Error] Failed to add interaction: {e}")
            # Still add the interaction even if priority calculation fails
            interaction.priority_score = 0.5  # Default middle priority
            interaction.priority_factors = {}
            self.interactions.append(interaction)
    
    def _update_priorities(self):
        """Periodically update priorities of all memories"""
        current_time = datetime.now()
        if (current_time - self.last_priority_update).total_seconds() > 300:  # Every 5 minutes
            print("[Debug] Updating all memory priorities...")
            memory_context = self._get_memory_context()
            for interaction in self.interactions:
                priority_score, factors = self.priority_system.calculate_priority_with_factors(
                    interaction, memory_context
                )
                interaction.priority_score = priority_score
                interaction.priority_factors = factors
            self.last_priority_update = current_time
    
    def _enforce_limit(self):
        """Modified to better handle priority-based removal"""
        if len(self.interactions) > self.max_interactions:
            # Calculate retention scores for all interactions
            scored_interactions = [
                (i, self._calculate_retention_score(i)) 
                for i in self.interactions
            ]
            
            # Sort by retention score (lowest first)
            scored_interactions.sort(key=lambda x: x[1])
            
            # Keep track of important memories
            important_count = sum(1 for _, score in scored_interactions if score > 0.7)
            
            # Remove lowest scoring memories until within limit
            while len(self.interactions) > self.max_interactions:
                to_remove, score = scored_interactions.pop(0)
                # Don't remove if it's important and we're not over important limit
                if score > 0.7 and important_count <= self.max_interactions * 0.2:  # Keep 20% important
                    continue
                self.interactions.remove(to_remove)
                print(f"[Debug] Removed memory with score {score:.2f}")
                print(f"[Debug] Factors: {to_remove.priority_factors}")
    
    
    def _calculate_retention_score(self, interaction: Interaction) -> float:
        """Calculate score for memory retention"""
        age_hours = (datetime.now() - interaction.timestamp).total_seconds() / 3600
        recency_score = 1 / (1 + age_hours)  # Decay over time
        return (interaction.priority_score * 0.7) + (recency_score * 0.3)
    
    def _get_memory_context(self) -> Dict:
        """Provide context for priority calculation"""
        return {
            'recent_interactions': self.interactions[-10:],
            'key_memories': [i.to_dict() for i in self.interactions 
                           if i.priority_score > 0.7]
        }
    
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
    
    
    def to_list(self):
        return [interaction.to_dict() for interaction in self.interactions]
    
    def load_from_list(self, data: List[Dict]):
        print("[Debug] Starting to load memory with timestamps:")
        self.interactions = []
        loaded_count = 0
        max_to_load = self.max_interactions  # Get the actual limit we want
        print(f"[Debug] Attempting to load up to {max_to_load} interactions...")
        
        # Remove any accidental limiting of the data
        for entry in data:  # Remove the slice operation
            try:
                if loaded_count >= max_to_load:
                    break
                    
                interaction = Interaction.from_dict(entry)
                print(f"[Debug] Loaded interaction timestamp: {interaction.timestamp}")
                self.interactions.append(interaction)
                loaded_count += 1
                
                if loaded_count % 100 == 0:
                    print(f"[Debug] Loaded {loaded_count} interactions so far...")
                    
            except Exception as e:
                print(f"[Debug] Error loading interaction: {e}")
                continue
                
        print(f"[Debug] Successfully loaded {len(self.interactions)} out of {len(data)} available interactions.")
        print(f"[Debug] Max interactions setting: {self.max_interactions}")


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

class ConsolidatedMemory:
    def __init__(self):
        self.patterns = {
            'conversation_patterns': {},  # Time patterns, style preferences
            'topic_patterns': {},        # Recurring topics and associations
            'emotional_patterns': {}     # Emotional response history
        }
        self.insights = []  # Generated insights about interactions
        self.key_memories = []  # Important memorable moments
        self.relationship_data = {
            'familiarity_level': 0,     # 0-100 scale
            'interaction_quality': [],   # List of scores
            'shared_interests': set(),   # Topics frequently discussed
            'conversation_style_preferences': {},
            'inside_references': {       # Track shared context and references
                'phrases': {},           # Memorable phrases or jokes
                'topics': {},            # Topic-specific shared understanding
                'context': {}            # Shared background knowledge
            }
        }
        self.last_consolidated = datetime.now()
    
    def to_dict(self):
        """Enhanced serialization for all memory aspects"""
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(i) for i in list(obj)]
            elif isinstance(obj, set):
                return list(obj)
            return obj

        return {
            'patterns': convert_datetime(self.patterns),
            'insights': self.insights,
            'key_memories': convert_datetime(self.key_memories),
            'relationship_data': {
                'familiarity_level': self.relationship_data['familiarity_level'],
                'interaction_quality': self.relationship_data['interaction_quality'],
                'shared_interests': list(self.relationship_data['shared_interests']),
                'conversation_style_preferences': self.relationship_data['conversation_style_preferences'],
                'inside_references': convert_datetime(self.relationship_data['inside_references'])
            },
            'last_consolidated': self.last_consolidated.isoformat()
        }

    @staticmethod
    def from_dict(data: Dict):
        memory = ConsolidatedMemory()
        memory.patterns = data.get('patterns', {})
        memory.insights = data.get('insights', [])
        memory.key_memories = data.get('key_memories', [])
        memory.relationship_data = {
            'familiarity_level': data['relationship_data'].get('familiarity_level', 0),
            'interaction_quality': data['relationship_data'].get('interaction_quality', []),
            'shared_interests': set(data['relationship_data'].get('shared_interests', [])),
            'conversation_style_preferences': data['relationship_data'].get('conversation_style_preferences', {}),
            'inside_references': data['relationship_data'].get('inside_references', {
                'phrases': {},
                'topics': {},
                'context': {}
            })
        }
        memory.last_consolidated = datetime.fromisoformat(data.get('last_consolidated', datetime.now().isoformat()))
        return memory

class MemoryConsolidator:
    def __init__(self, short_memory: ShortTermMemory, long_memory: LongTermMemory):
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.consolidated_memory = ConsolidatedMemory()
        self.consolidation_interval = timedelta(seconds=30)
        self.emotion_weights = {
            'joy': 1.5,
            'sadness': 1.3,
            'anger': 1.4,
            'surprise': 1.2,
            'neutral': 1.0
        }

    def should_consolidate(self) -> bool:
        """Check if enough time has passed for consolidation"""
        time_since_last = datetime.now() - self.consolidated_memory.last_consolidated
        return time_since_last >= self.consolidation_interval

    def save_consolidated_memory(self, file_path: str = "consolidated_memory.json"):
        """Save consolidated memory to file"""
        with open(file_path, 'w') as f:
            json.dump(self.consolidated_memory.to_dict(), f, indent=4)

    def load_consolidated_memory(self, file_path: str = "consolidated_memory.json"):
        """Load consolidated memory from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.consolidated_memory = ConsolidatedMemory.from_dict(data)
        except FileNotFoundError:
            self.consolidated_memory = ConsolidatedMemory()

    def _analyze_conversation_patterns(self, interactions: List[Interaction]) -> Dict:
        """Analyze patterns in conversation flow and timing"""
        patterns = {
            'response_times': [],
            'conversation_length': [],
            'topic_transitions': []
        }
        
        for i in range(1, len(interactions)):
            time_diff = (interactions[i].timestamp - 
                        interactions[i-1].timestamp).total_seconds()
            patterns['response_times'].append(time_diff)
            
            # Add more pattern analysis as needed
            
        return dict(patterns)

    def _analyze_emotional_patterns(self, interactions: List[Interaction]) -> Dict:
        """Analyze emotional content and progression"""
        patterns = {
            'emotions': [],
            'emotional_intensity': {},
            'emotional_progression': []
        }
        
        for interaction in interactions:
            emotion = self._detect_emotion(interaction.user_message)
            patterns['emotions'].append({
                'timestamp': interaction.timestamp.isoformat(),
                'emotion': emotion,
                'intensity': self.emotion_weights.get(emotion, 1.0)
            })
            
        return dict(patterns)

    def _analyze_topic_patterns(self, interactions: List[Interaction]) -> Dict:
        """Analyze patterns in conversation topics"""
        patterns = {
            'topic_frequency': defaultdict(int),  # Use defaultdict to avoid KeyError
            'topic_chains': [],
            'topic_duration': {}
        }
        
        for interaction in interactions:
            for tag in interaction.tags:
                patterns['topic_frequency'][tag] += 1
                
        return dict(patterns)

    def consolidate_memories(self) -> ConsolidatedMemory:
        """Main consolidation process"""
        if not self.should_consolidate():
            return self.consolidated_memory

        print("[Debug] Starting memory consolidation...")
        recent_interactions = self.long_memory.interactions[-50:]

        # Analyze patterns
        conversation_patterns = self._analyze_conversation_patterns(recent_interactions)
        emotional_patterns = self._analyze_emotional_patterns(recent_interactions)
        topic_patterns = self._analyze_topic_patterns(recent_interactions)
        
        # Update consolidated memory
        self.consolidated_memory.patterns.update({
            'conversation_patterns': conversation_patterns,
            'emotional_patterns': emotional_patterns,
            'topic_patterns': topic_patterns
        })
        
        # Generate and update insights
        new_insights = self._generate_insights(conversation_patterns, emotional_patterns, topic_patterns)
        self.consolidated_memory.insights.extend(new_insights)
        if len(self.consolidated_memory.insights) > 20:
            self.consolidated_memory.insights = self.consolidated_memory.insights[-20:]
        
        # Update relationship data
        self._update_relationship_data(recent_interactions)
        
        # Update timestamp and save
        self.consolidated_memory.last_consolidated = datetime.now()
        self.save_consolidated_memory()
        
        print("[Debug] Memory consolidation complete and saved")
        return self.consolidated_memory

    def _generate_insights(self, conversation_patterns: Dict,
                        emotional_patterns: Dict,
                        topic_patterns: Dict) -> List[str]:
        """Generate insights based on analyzed patterns"""
        insights = []
        
        # Analyze conversation patterns
        if 'response_times' in conversation_patterns and conversation_patterns['response_times']:
            avg_time = statistics.mean(conversation_patterns['response_times'])
            if avg_time < 30:
                insights.append("Conversation has quick response times")
            elif avg_time < 120:
                insights.append("Conversation has moderate response times")
            else:
                insights.append("Conversation has relaxed response times")
        
        # Analyze emotional patterns
        if 'emotions' in emotional_patterns:
            emotions = [e['emotion'] for e in emotional_patterns['emotions']]
            if emotions:
                dominant = max(set(emotions), key=emotions.count)
                insights.append(f"Conversation shows primarily {dominant} emotions")
        
        # Analyze topic patterns
        if 'topic_frequency' in topic_patterns:
            frequent_topics = sorted(
                topic_patterns['topic_frequency'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            if frequent_topics:
                topics = ", ".join(topic for topic, _ in frequent_topics)
                insights.append(f"Most discussed topics: {topics}")
        
        return insights

    def _update_relationship_data(self, interactions: List[Interaction]):
        """Update relationship metrics"""
        if not interactions:
            return
            
        # Update familiarity level
        interaction_count = len(self.long_memory.interactions)
        self.consolidated_memory.relationship_data['familiarity_level'] = min(
            100, int(interaction_count / 10)
        )
        
        # Update shared interests
        for interaction in interactions:
            self.consolidated_memory.relationship_data['shared_interests'].update(
                interaction.tags
            )

    def _detect_emotion(self, text: str) -> str:
        """Reuse emotion detection from MemoryPriority"""
        # Implementation similar to MemoryPriority._detect_emotion
        return 'neutral'  # Default fallback

class PersonalityManager:
    def __init__(self):
        self.default_personality_name = "default"
        
    def get_personality_files(self, name: str) -> Dict[str, str]:
        """Get file paths for a given personality name"""
        if name == self.default_personality_name:
            # Use existing default file names
            return {
                "personality": "personality.json",
                "short_memory": "short_memory.json",
                "long_memory": "long_memory.json",
                "consolidated_memory": "consolidated_memory.json"
            }
        else:
            # Use name-specific files
            base_name = name.lower()
            return {
                "personality": f"{base_name}_personality.json",
                "short_memory": f"{base_name}_short_term_memory.json",
                "long_memory": f"{base_name}_long_term_memory.json",
                "consolidated_memory": f"{base_name}_consolidated_memory.json"
            }
    
    def create_default_personality(self, name: str) -> PersonalityProfile:
        """Create default personality settings with specified name"""
        profile = PersonalityProfile(
            tone="neutral",
            response_style="balanced",
            behavior="reactive",
            user_preferences={}
        )
        profile.set_name(name)
        return profile
    
    def load_or_create_personality(self, name: str) -> Tuple[PersonalityProfile, ShortTermMemory, LongTermMemory]:
        """Load existing personality or create new one"""
        files = self.get_personality_files(name)
        
        # Try to load existing personality
        try:
            personality = PersonalityProfile.load_from_file(files["personality"])
            personality.set_name(name)  # Ensure name is set
            print(f"[Info] Loaded existing personality: {name}")
        except FileNotFoundError:
            # Create new personality
            personality = self.create_default_personality(name)
            personality.save_to_file(files["personality"])
            print(f"[Info] Created new personality: {name}")
        
        # Initialize memories
        short_memory = ShortTermMemory(max_interactions=10)
        long_memory = LongTermMemory(max_interactions=1000)
        
        # Try to load existing memories
        try:
            with open(files["short_memory"], "r") as f:
                short_memory.load_from_list(json.load(f))
        except FileNotFoundError:
            print(f"[Info] No existing short-term memory for {name}. Starting fresh.")
        
        try:
            with open(files["long_memory"], "r") as f:
                long_memory.load_from_list(json.load(f))
        except FileNotFoundError:
            print(f"[Info] No existing long-term memory for {name}. Starting fresh.")
        
        return personality, short_memory, long_memory
    
    def save_personality_state(self, name: str, personality: PersonalityProfile, 
                            short_memory: ShortTermMemory, long_memory: LongTermMemory,
                            consolidated_memory: ConsolidatedMemory):
        """Save all personality-related files"""
        files = self.get_personality_files(name)
        
        try:
            personality.save_to_file(files["personality"])
            
            with open(files["short_memory"], "w") as f:
                json.dump(short_memory.to_list(), f, indent=4)
                
            with open(files["long_memory"], "w") as f:
                json.dump(long_memory.to_list(), f, indent=4)
                
            with open(files["consolidated_memory"], "w") as f:
                json.dump(consolidated_memory.to_dict(), f, indent=4)
            
            print(f"[Debug] Saved all state files for personality: {name}")
        except Exception as e:
            print(f"[Error] Failed to save personality state: {e}")
    
    def list_available_personalities(self) -> List[str]:
        """List all available personalities including default"""
        personalities = set()
        
        # Check for default personality
        if os.path.exists("personality.json"):
            personalities.add("default")
            
        # Check for other personalities
        for file in os.listdir():
            if file.endswith("_personality.json"):
                name = file.replace("_personality.json", "")
                personalities.add(name)
                
        return sorted(list(personalities))

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
        self.memory_consolidator = MemoryConsolidator(short_memory, long_memory)  # Add this line
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

        # Add this line to trigger memory consolidation
        self.memory_consolidator.consolidate_memories()
        
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
            f"You are {self.personality.name}, an artificial personality with the following characteristics:\n"
            f"Tone: {self.personality.tone}\n"
            f"Response Style: {self.personality.response_style}\n"
            f"Behavior: {self.personality.behavior}\n\n"
            f"Current time: {datetime.now()}\n" 
            "Time Context (BE PRECISE WITH THESE TIMES):\n"  # Added emphasis on precision
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
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
    
    # Initialize personality manager
    personality_manager = PersonalityManager()
    
    # Check available personalities
    available_personalities = personality_manager.list_available_personalities()
    
    # Display prompt
    print("\nWelcome! Please choose a personality to interact with.")
    print("Press Enter to use the default personality, or type a name to use/create a new personality.")
    
    if available_personalities:
        print("\nAvailable personalities:")
        for name in available_personalities:
            if name == "default":
                print(f"- default (press Enter to select)")
            else:
                print(f"- {name}")
    
    # Prompt for personality selection
    while True:
        personality_name = input("\nEnter personality name: ").strip().lower()
        if not personality_name:
            personality_name = "default"
            print("Using default personality...")
        else:
            print(f"Using/creating personality: {personality_name}")
        
        if personality_name.lower() == "exit":
            print("Goodbye!")
            return
            
        # Load or create personality and memories
        try:
            personality, short_memory, long_memory = personality_manager.load_or_create_personality(personality_name)
            break
        except Exception as e:
            print(f"Error loading personality: {e}")
            print("Please try again or type 'exit' to quit.")
    
    # Initialize chatbot with selected personality
    try:
        chatbot = ChatBot(personality, short_memory, long_memory)
    except ValueError as e:
        print(f"[Error] {e}")
        print("Please ensure that the API_KEY is set in the .env file.")
        return
    
    print(f"\nChatBot is ready with personality: {personality_name}")
    print("Type your messages below. Type 'exit' or 'bye' to end the conversation.")
    print("Type 'debug' to see memory status.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
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
    
    # Save personality state before exiting
    personality_manager.save_personality_state(
    personality_name, 
    personality, 
    short_memory, 
    long_memory,
    chatbot.memory_consolidator.consolidated_memory)
    print(f"Session saved for personality: {personality_name}. Goodbye!")

if __name__ == "__main__":
    main()