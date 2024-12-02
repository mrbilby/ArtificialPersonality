import json
from openai import OpenAI  # As per your working setup
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import statistics
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import networkx as nx

# Load environment variables from .env file
load_dotenv()

def normalize_timestamp(timestamp_str):
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        if timestamp > datetime.now():
            print(f"[Warning] Future timestamp detected: {timestamp_str}. Adjusting to current time.")
            return datetime.now()
        return timestamp
    except ValueError:
        print(f"[Warning] Invalid timestamp format: {timestamp_str}. Setting to current time.")
        return datetime.now()


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
        
        # First try to load existing data
        existing_data = {}
        try:
            with open(save_path, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            pass  # No existing file, will create new one
            
        # Update with new data, preserving existing values if not changed
        data = {
            "tone": self.tone,
            "response_style": self.response_style,
            "behavior": self.behavior,
            "user_preferences": self.user_preferences,
            "do_dont": self.do_dont,
            "name": existing_data.get("name", self.name)  # Preserve existing name if present
        }
        
        with open(save_path, "w") as file:
            json.dump(data, file, indent=4)
        print(f"[Debug] Personality profile updated in {save_path}")

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
        """Convert the interaction to a dictionary for storage."""
        return {
            "user_message": self.user_message,
            "bot_response": self.bot_response,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),  # Ensure ISO format
            "priority_score": self.priority_score,
            "priority_factors": self.priority_factors,
        }

    @staticmethod
    def from_dict(data: Dict):
        """Recreate an Interaction object from a dictionary."""
        interaction = Interaction(
            user_message=data["user_message"],
            bot_response=data["bot_response"],
            tags=data.get("tags", [])
        )
        # Deserialize timestamp correctly
        timestamp_str = data.get("timestamp")
        if timestamp_str:
            try:
                interaction.timestamp = datetime.fromisoformat(timestamp_str)
                if interaction.timestamp > datetime.now():
                    print(f"[Warning] Future timestamp detected: {interaction.timestamp}. Adjusting to current time.")
                    interaction.timestamp = datetime.now()
            except ValueError:
                print(f"[Warning] Invalid timestamp format: {timestamp_str}. Setting to current time.")
                interaction.timestamp = datetime.now()
        else:
            print(f"[Warning] Missing timestamp in interaction. Setting to current time.")
            interaction.timestamp = datetime.now()
        interaction.priority_score = float(data.get("priority_score", 0.0))
        interaction.priority_factors = data.get("priority_factors", {})
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
        """Updated priority calculation using new emotion detection"""
        factors = {}
        priority_score = 0.0
        
        # Emotional Impact (using new emotion detection)
        emotion, emotional_score = self._detect_emotion(interaction.user_message)
        factors['emotional_impact'] = emotional_score
        priority_score += emotional_score * 0.35
        
        # Recency (same as before)
        time_diff = (datetime.now() - interaction.timestamp).total_seconds()
        recency_score = 1 / (1 + (time_diff / (24 * 3600)))
        factors['recency'] = recency_score
        priority_score += recency_score * 0.15
        
        # User Importance (same as before)
        importance_score = self._calculate_user_importance(interaction)
        factors['user_importance'] = importance_score
        priority_score += importance_score * 0.30
        
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

    def _detect_emotion(self, text: str) -> tuple[str, float]:
        """
        Detect dominant emotion and its intensity in text.
        Returns tuple of (emotion, intensity)
        """
        # Expanded emotion keywords with intensity weights
        emotion_keywords = {
            'joy': {
                'high': {
                    'overjoyed': 0.9, 'ecstatic': 0.9, 'thrilled': 0.9, 'fantastic': 0.8, 'amazing': 0.8,
                    'happiest': 0.9, 'best': 0.8  # Added for phrases like "happiest day"
                },
                'medium': {
                    'happy': 0.6, 'glad': 0.6, 'pleased': 0.6, 'good': 0.5,
                    'great': 0.6, 'wonderful': 0.6
                },
                'low': {
                    'content': 0.4, 'nice': 0.4, 'okay': 0.3
                }
            },
            'sadness': {
                'high': {
                    'devastated': 0.9, 'heartbroken': 0.9, 'miserable': 0.9, 'depressed': 0.8
                },
                'medium': {
                    'sad': 0.6, 'unhappy': 0.6, 'down': 0.6, 'upset': 0.5
                },
                'low': {
                    'disappointed': 0.4, 'blue': 0.4, 'meh': 0.3
                }
            },
            'anger': {
                'high': {
                    'furious': 0.9, 'enraged': 0.9, 'livid': 0.9, 'outraged': 0.8
                },
                'medium': {
                    'angry': 0.6, 'mad': 0.6, 'irritated': 0.6, 'annoyed': 0.5
                },
                'low': {
                    'frustrated': 0.4, 'bothered': 0.4, 'displeased': 0.3
                }
            },
            'surprise': {
                'high': {
                    'shocked': 0.9, 'astonished': 0.9, 'astounded': 0.9, 'stunned': 0.8,
                    'wow': 0.9, "can't believe": 0.9  # Added for common surprise expressions
                },
                'medium': {
                    'surprised': 0.6, 'amazed': 0.6, 'unexpected': 0.6,
                    'unbelievable': 0.7
                },
                'low': {
                    'curious': 0.4, 'interesting': 0.4, 'different': 0.3
                }
            }
        }

        intensity_modifiers = {
            'increase': {'very': 0.15, 'really': 0.15, 'so': 0.15, 'extremely': 0.2, 'absolutely': 0.2, 'completely': 0.2},
            'decrease': {'slightly': -0.15, 'somewhat': -0.15, 'a bit': -0.15, 'kind of': -0.1, 'sort of': -0.1}
        }

        text_lower = text.lower()
        words = text_lower.split()
        
        # Initialize emotion scores
        emotion_scores = {emotion: 0.0 for emotion in emotion_keywords.keys()}
        max_intensity = 0.0
        
        # First pass: Find base emotions and their intensities
        for emotion, levels in emotion_keywords.items():
            for level, words_dict in levels.items():
                for word, intensity in words_dict.items():
                    if word in text_lower:
                        current_intensity = intensity
                        
                        # Look for intensity modifiers before the emotion word
                        word_index = text_lower.find(word)
                        preceding_text = text_lower[:word_index]
                        
                        # Check for intensity modifiers
                        for modifier_type, modifiers in intensity_modifiers.items():
                            for modifier, mod_value in modifiers.items():
                                if modifier in preceding_text:
                                    if modifier_type == 'increase':
                                        current_intensity = min(1.0, current_intensity + mod_value)
                                    else:
                                        current_intensity = max(0.1, current_intensity + mod_value)
                        
                        emotion_scores[emotion] = max(emotion_scores[emotion], current_intensity)
                        max_intensity = max(max_intensity, current_intensity)

        # Additional context analysis
        if '!' in text:
            # Check for surprise indicators with exclamation
            if any(word in text_lower for word in ['wow', 'whoa', "can't believe", 'oh']):
                emotion_scores['surprise'] = max(emotion_scores['surprise'], 0.8)
        
        # If no emotion is detected, return neutral with base intensity
        if max(emotion_scores.values()) == 0:
            return ('neutral', 0.3)
        
        # Find the highest scoring emotion
        max_score = max(emotion_scores.values())
        tied_emotions = [
            emotion for emotion, score in emotion_scores.items() 
            if score == max_score
        ]
        
        # Handle ties
        if len(tied_emotions) > 1:
            if '!' in text:  # Fixed: Removed any()
                for emotion in ['joy', 'anger', 'surprise']:
                    if emotion in tied_emotions:
                        return (emotion, max_score)
            if '?' in text:
                if 'surprise' in tied_emotions:
                    return ('surprise', max_score)
            
            # If still tied, return the first one
            return (tied_emotions[0], max_score)
        
        # Return the single highest scoring emotion
        dominant_emotion = tied_emotions[0]
        return (dominant_emotion, max_score)  
      
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
        try:
            sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        except Exception as e:
            print(f"[Debug] Error sorting interactions: {e}")
            return patterns  # Early return since we can't proceed without sorted interactions

        # Proceed with analysis
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


class GraphMemoryManager:
    def __init__(self, personality_name='default'):
        self.graph_file = self._get_graph_file(personality_name)
        self.G = self._load_or_create_graph()

    def _get_graph_file(self, personality_name: str) -> str:
        """Determine the graph file based on the personality name."""
        if personality_name.lower() == 'default':
            return 'memory_graph.json'
        else:
            return f'{personality_name.lower()}_memory_graph.json'


    def _load_or_create_graph(self):
        """Load existing graph or create a new one."""
        try:
            with open(self.graph_file, 'r') as f:
                data = json.load(f)
                G = nx.Graph()
                
                # Reconstruct nodes
                for node_id, node_data in data['nodes']:
                    # Ensure 'timestamp' and 'tags' exist
                    if 'timestamp' not in node_data or 'tags' not in node_data:
                        print(f"[Warning] Node {node_id} missing 'timestamp' or 'tags'. Skipping node.")
                        continue
                    G.add_node(node_id, **node_data)
                
                # Reconstruct edges
                for u, v, data in data['edges']:
                    G.add_edge(u, v, **data)
                
                return G
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"[Debug] No existing graph found for {self.graph_file}. Creating a new graph.")
            return nx.Graph()


    def save_graph(self):
        """Save graph to file."""
        graph_data = {
            'nodes': [[n, data] for n, data in self.G.nodes(data=True)],
            'edges': [[u, v, data] for u, v, data in self.G.edges(data=True)]
        }
        with open(self.graph_file, 'w') as f:
            json.dump(graph_data, f, indent=4)
        print(f"[Debug] Memory graph saved to {self.graph_file}")


    def calculate_tag_similarity(self, tags1, tags2):
        """Calculate Jaccard similarity between two sets of tags."""
        if not tags1 or not tags2:
            return 0
        set1 = set(tags1)
        set2 = set(tags2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0

    def calculate_temporal_proximity(self, time1, time2, max_time_diff=86400):
        """Calculate temporal proximity between two timestamps."""
        try:
            t1 = datetime.fromisoformat(time1)
            t2 = datetime.fromisoformat(time2)
            time_diff = abs((t1 - t2).total_seconds())
            return max(0, 1 - (time_diff / max_time_diff))
        except Exception as e:
            print(f"[Warning] Error calculating temporal proximity: {e}")
            return 0  # Default to 0 if calculation fails


    def add_memory(self, memory):
        """Add a new memory to the graph."""
        # Validate timestamp and tags
        if not memory.get("timestamp") or not memory.get("tags"):
            print(f"[Error] Memory missing required fields: {memory}")
            return
        
        try:
            # Validate and normalize timestamp
            if isinstance(memory['timestamp'], str):
                timestamp = datetime.fromisoformat(memory['timestamp'])
                if timestamp.year > datetime.now().year + 1:  # Allow for small clock differences
                    print(f"[Error] Invalid future timestamp detected: {timestamp}")
                    timestamp = datetime.now()
                memory['timestamp'] = timestamp.isoformat()
            
            node_id = max(self.G.nodes(), default=-1) + 1
            self.G.add_node(node_id,
                            message=memory['user_message'],
                            response=memory['bot_response'],
                            timestamp=memory['timestamp'],
                            tags=memory['tags'],
                            priority=memory.get('priority_score', 0.5))
            
            # Add edges to other nodes based on similarity
            for existing_id in list(self.G.nodes):
                if existing_id != node_id:
                    existing_node = self.G.nodes[existing_id]
                    # Check if 'tags' and 'timestamp' exist
                    if "tags" not in existing_node or "timestamp" not in existing_node:
                        print(f"[Warning] Node {existing_id} missing 'tags' or 'timestamp'. Skipping.")
                        continue  # Skip nodes with missing data
                    
                    tag_sim = self.calculate_tag_similarity(
                        memory["tags"], existing_node["tags"]
                    )
                    temp_prox = self.calculate_temporal_proximity(
                        memory["timestamp"], existing_node["timestamp"]
                    )
                    edge_weight = (0.7 * tag_sim) + (0.3 * temp_prox)
                    if edge_weight > 0.1:
                        self.G.add_edge(node_id, existing_id, weight=edge_weight)
            
            # Save the graph
            self.save_graph()
            print(f"[Debug] Memory added to graph: {memory['user_message'][:30]}...")
        except Exception as e:
            print(f"[Error] Failed to add memory: {e}")


    def find_similar_memories(self, query_tags, top_n=5):
        """Find most similar memories to query tags."""
        similarities = []
        
        for node in self.G.nodes():
            node_tags = self.G.nodes[node]['tags']
            sim = self.calculate_tag_similarity(query_tags, node_tags)
            # Add node centrality to boost important memories
            centrality = nx.degree_centrality(self.G)[node]
            combined_score = (0.8 * sim) + (0.2 * centrality)
            similarities.append((node, combined_score))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]


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
    
    def retrieve_relevant_interactions(self, query: str, top_n=15) -> List[Interaction]:
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
        """Load consolidated memory from file with proper max_interactions"""
        print("[Debug] Starting to load memory with timestamps:")
        self.interactions = []
        loaded_count = 0
        print(f"[Debug] Attempting to load up to {self.max_interactions} interactions...")
        
        for entry in data:
            try:
                if loaded_count >= self.max_interactions:
                    break
                interaction = Interaction.from_dict(entry)
                # Validate the timestamp
                if not interaction.timestamp or interaction.timestamp > datetime.now():
                    interaction.timestamp = datetime.now()
                self.interactions.append(interaction)
                loaded_count += 1
            except Exception as e:
                print(f"[Debug] Error loading interaction: {e}")
                continue

                
        print(f"[Debug] Successfully loaded {len(self.interactions)} out of {len(data)} available interactions.")
        print(f"[Debug] Max interactions setting: {self.max_interactions}")


class ShortTermMemory(Memory):
    def retrieve_relevant_interactions(self, query: str, top_n=None) -> List[Interaction]:
        """Modified to use class max_interactions if top_n not specified"""
        if top_n is None:
            top_n = self.max_interactions

        # Get all interactions, sorted by timestamp (most recent first)
        all_interactions = sorted(self.interactions, key=lambda x: x.timestamp, reverse=True)
        
        # Return up to max_interactions most recent ones
        return all_interactions[:top_n]

    def _enforce_limit(self):
        """Ensure we keep the most recent interactions up to max_interactions"""
        if len(self.interactions) > self.max_interactions:
            # Sort by timestamp and keep most recent
            self.interactions.sort(key=lambda x: x.timestamp, reverse=True)
            self.interactions = self.interactions[:self.max_interactions]

class LongTermMemory(Memory):
    def __init__(self, max_interactions: int = 1000, personality_name: str = 'default'):
        super().__init__(max_interactions)
        self.graph_manager = GraphMemoryManager(personality_name)

    def add_interaction(self, interaction: Interaction):
        # Call parent class add_interaction to maintain core functionality
        super().add_interaction(interaction)

        # Add to graph after successful addition to main memory
        try:
            memory_dict = interaction.to_dict()
            self.graph_manager.add_memory(memory_dict)
            self.graph_manager.save_graph()  # Save graph to persist changes
            print("[Debug] Interaction added to memory graph and saved.")
        except Exception as e:
            print(f"[Error] Failed to add interaction to graph: {e}")

    def retrieve_relevant_interactions(self, query: str, top_n=15) -> List[Interaction]:
        """Enhanced retrieval combining graph search with traditional method"""
        try:
            # Extract tags from query
            query_tags = self._extract_query_tags(query)
            
            # Get similar memories from graph
            similar_nodes = self.graph_manager.find_similar_memories(query_tags, top_n)
            
            # Get corresponding interactions from graph matches
            graph_relevant = []
            for node_id, score in similar_nodes:
                node_data = self.graph_manager.G.nodes[node_id]
                # Find matching interaction
                for interaction in self.interactions:
                    if (interaction.user_message == node_data['message'] and 
                        interaction.timestamp == node_data['timestamp']):
                        graph_relevant.append(interaction)
                        break

            # Get traditional matches using parent class method
            traditional_relevant = super().retrieve_relevant_interactions(query, top_n)
            
            # Combine results while maintaining uniqueness
            combined = []
            seen = set()
            
            # Prioritize graph matches
            for interaction in graph_relevant:
                if interaction not in seen:
                    combined.append(interaction)
                    seen.add(interaction)
            
            # Add traditional matches
            for interaction in traditional_relevant:
                if interaction not in seen:
                    combined.append(interaction)
                    seen.add(interaction)
            
            print(f"[Debug] Retrieved {len(combined)} total interactions "
                  f"({len(graph_relevant)} from graph, {len(traditional_relevant)} from traditional)")
            
            return combined[:top_n]
            
        except Exception as e:
            print(f"[Error] Failed to retrieve interactions using graph: {e}")
            # Fallback to traditional method
            return super().retrieve_relevant_interactions(query, top_n)

    def _extract_query_tags(self, query: str) -> List[str]:
        """Extract tags from query text for similarity matching"""
        words = query.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are'}
        return [word for word in words if word not in stop_words and len(word) > 3]


    def retrieve_relevant_interactions_by_tags(self, tags: List[str], top_n=15):
        """Enhanced tag-based retrieval using graph and traditional methods."""
        try:
            if not tags:
                print("[Error] No tags provided for retrieval.")
                return []

            print(f"[Debug] Searching for interactions matching tags: {tags}")
            
            # Step 1: Graph-based retrieval
            similar_nodes = self.graph_manager.find_similar_memories(tags, top_n)
            graph_relevant = []
            
            for node_id, score in similar_nodes:
                node_data = self.graph_manager.G.nodes[node_id]
                # Find the corresponding interaction
                for interaction in self.interactions:
                    if (interaction.user_message == node_data.get('message') and 
                        interaction.timestamp == node_data.get('timestamp')):
                        graph_relevant.append(interaction)
                        break
            
            print(f"[Debug] Graph-based retrieval found {len(graph_relevant)} interactions.")

            # Step 2: Fallback to traditional retrieval if needed
            if len(graph_relevant) < top_n:
                relevant_scores = []
                search_tags = [tag.lower() for tag in tags]  # Normalize tags for comparison
                
                for interaction in self.interactions:
                    score = 0
                    interaction_text = f"{interaction.user_message} {interaction.bot_response}".lower()
                    
                    # 2.1: Exact tag matches (highest weight)
                    for tag in search_tags:
                        if tag in interaction.tags:
                            score += 2
                    
                    # 2.2: Partial tag matches (lower weight)
                    for tag in search_tags:
                        for interaction_tag in interaction.tags:
                            if (tag in interaction_tag or interaction_tag in tag) and tag != interaction_tag:
                                score += 1
                    
                    # 2.3: Content-based matches (lowest weight)
                    for tag in search_tags:
                        if tag in interaction_text:
                            score += 0.5
                    
                    # 2.4: Time decay factor (recent interactions prioritized)
                    time_diff = (datetime.now() - interaction.timestamp).total_seconds() / (24 * 3600)  # Days
                    time_factor = 1 / (1 + time_diff)
                    
                    # Combine scores with priority adjustment
                    final_score = score * (0.7 + 0.3 * time_factor)
                    if hasattr(interaction, 'priority_score'):
                        final_score *= (1 + interaction.priority_score)
                    
                    if final_score > 0:
                        relevant_scores.append((interaction, final_score))
                
                # Sort by relevance score
                relevant_scores.sort(key=lambda x: x[1], reverse=True)
                additional_interactions = [interaction for interaction, score 
                                           in relevant_scores[:top_n - len(graph_relevant)]
                                           if interaction not in graph_relevant]
                graph_relevant.extend(additional_interactions)
            
            print(f"[Debug] Total relevant interactions found: {len(graph_relevant)}")
            return graph_relevant[:top_n]
        
        except Exception as e:
            print(f"[Error] Failed to retrieve interactions by tags: {e}")
            return []

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
    
    def is_empty(self) -> bool:
        """Check if the consolidated memory has any meaningful data."""
        return not (
            self.patterns.get('conversation_patterns') or
            self.patterns.get('topic_patterns') or
            self.patterns.get('emotional_patterns') or
            self.insights or
            self.key_memories or
            self.relationship_data.get('shared_interests') or
            self.relationship_data.get('interaction_quality') or
            self.relationship_data.get('conversation_style_preferences') or
            self.relationship_data.get('inside_references', {}).get('phrases') or
            self.relationship_data.get('inside_references', {}).get('topics') or
            self.relationship_data.get('inside_references', {}).get('context')
        )
    

    @staticmethod
    def from_dict(data: Dict):
        memory = ConsolidatedMemory()
        memory.patterns = data.get('patterns', {})
        memory.insights = data.get('insights', [])
        memory.key_memories = data.get('key_memories', [])
        
        # Safely get 'relationship_data' and its subfields
        relationship_data = data.get('relationship_data', {})
        memory.relationship_data = {
            'familiarity_level': relationship_data.get('familiarity_level', 0),
            'interaction_quality': relationship_data.get('interaction_quality', []),
            'shared_interests': set(relationship_data.get('shared_interests', [])),
            'conversation_style_preferences': relationship_data.get('conversation_style_preferences', {}),
            'inside_references': relationship_data.get('inside_references', {
                'phrases': {},
                'topics': {},
                'context': {}
            })
        }

        # Safely parse 'last_consolidated' timestamp
        last_consolidated_str = data.get('last_consolidated')
        if last_consolidated_str:
            try:
                memory.last_consolidated = datetime.fromisoformat(last_consolidated_str)
            except ValueError:
                print(f"[Error] Invalid 'last_consolidated' format: {last_consolidated_str}")
                memory.last_consolidated = datetime.now()
        else:
            memory.last_consolidated = datetime.now()

        return memory
    

class MemoryConsolidator:
    def __init__(self, short_memory: ShortTermMemory, long_memory: LongTermMemory, personality_name: str = 'default'):
        self.short_memory = short_memory
        self.long_memory = long_memory
        self.personality_name = personality_name
        self.consolidated_memory = ConsolidatedMemory()
        self.consolidation_interval = timedelta(seconds=30)
        self.emotion_weights = {
            'joy': 1.5,
            'sadness': 1.3,
            'anger': 1.4,
            'surprise': 1.2,
            'neutral': 1.0
        }
        # Add this line to create an instance of MemoryPriority
        self.priority_system = MemoryPriority()

    def _detect_emotion(self, text: str) -> str:
        """Use MemoryPriority's emotion detection"""
        emotion, _ = self.priority_system._detect_emotion(text)
        return emotion

    def load_consolidated_memory(self, file_path: str = None):
        """Load consolidated memory from file."""
        if file_path is None:
            if self.personality_name == 'default':
                file_path = "consolidated_memory.json"
            else:
                file_path = f"{self.personality_name}_consolidated_memory.json"
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.consolidated_memory = ConsolidatedMemory.from_dict(data)
                print(f"[Debug] Loaded consolidated memory from {file_path}")
        except FileNotFoundError:
            print(f"[Debug] No existing consolidated memory found at {file_path}. Starting fresh.")
            if not hasattr(self, 'consolidated_memory') or self.consolidated_memory is None:
                self.consolidated_memory = ConsolidatedMemory()
        except Exception as e:
            print(f"[Error] Failed to load consolidated memory: {e}")
            # Do not overwrite existing consolidated_memory if already set
            if not hasattr(self, 'consolidated_memory') or self.consolidated_memory is None:
                self.consolidated_memory = ConsolidatedMemory()

    def save_consolidated_memory(self, file_path: str = None):
        """Save consolidated memory to file."""
        if file_path is None:
            if self.personality_name == 'default':
                file_path = "consolidated_memory.json"
            else:
                file_path = f"{self.personality_name}_consolidated_memory.json"
        try:
            # Check if consolidated_memory has meaningful data
            if not self.consolidated_memory or self.consolidated_memory.is_empty():
                print(f"[Debug] Consolidated memory is empty. Skipping save to avoid overwriting existing data.")
                return
            with open(file_path, 'w') as f:
                json.dump(self.consolidated_memory.to_dict(), f, indent=4)
            print(f"[Debug] Saved consolidated memory to {file_path}")
        except Exception as e:
            print(f"[Error] Failed to save consolidated memory: {e}")

    def should_consolidate(self) -> bool:
        """Check if enough time has passed for consolidation"""
        time_since_last = datetime.now() - self.consolidated_memory.last_consolidated
        return time_since_last >= self.consolidation_interval

    def consolidate_memories(self) -> ConsolidatedMemory:
        """Main consolidation process"""
        if not self.should_consolidate():
            return self.consolidated_memory

        print("[Debug] Starting memory consolidation...")
        recent_interactions = self.long_memory.interactions[-50:]

        # If no recent interactions, skip consolidation
        if not recent_interactions:
            print("[Debug] No recent interactions to consolidate. Skipping.")
            return self.consolidated_memory

        data_updated = False

        # Analyze patterns
        conversation_patterns = self._analyze_conversation_patterns(recent_interactions)
        emotional_patterns = self._analyze_emotional_patterns(recent_interactions)
        topic_patterns = self._analyze_topic_patterns(recent_interactions)
        
        # Update consolidated memory patterns only if there are new patterns
        if conversation_patterns:
            self.consolidated_memory.patterns['conversation_patterns'].update(conversation_patterns)
            data_updated = True
        if emotional_patterns:
            self.consolidated_memory.patterns['emotional_patterns'].update(emotional_patterns)
            data_updated = True
        if topic_patterns:
            self.consolidated_memory.patterns['topic_patterns'].update(topic_patterns)
            data_updated = True
        
        # Generate and update insights only if there are new insights
        new_insights = self._generate_insights(conversation_patterns, emotional_patterns, topic_patterns)
        if new_insights:
            self.consolidated_memory.insights.extend(new_insights)
            if len(self.consolidated_memory.insights) > 20:
                self.consolidated_memory.insights = self.consolidated_memory.insights[-20:]
            data_updated = True
        
        # Update relationship data
        if self._update_relationship_data(recent_interactions):
            data_updated = True
        
        if data_updated:
            # Update timestamp and save
            self.consolidated_memory.last_consolidated = datetime.now()
            self.save_consolidated_memory()
            print("[Debug] Memory consolidation complete and saved")
        else:
            print("[Debug] No new data to consolidate. Skipping save.")

        return self.consolidated_memory

    # Adjust _update_relationship_data to return True if data was updated
    def _update_relationship_data(self, interactions: List[Interaction]) -> bool:
        """Update relationship metrics"""
        data_updated = False
        if not interactions:
            return data_updated

        # Update familiarity level
        interaction_count = len(self.long_memory.interactions)
        new_familiarity_level = min(100, int(interaction_count / 10))
        if new_familiarity_level != self.consolidated_memory.relationship_data['familiarity_level']:
            self.consolidated_memory.relationship_data['familiarity_level'] = new_familiarity_level
            data_updated = True

        # Update shared interests
        new_shared_interests = set(self.consolidated_memory.relationship_data['shared_interests'])
        for interaction in interactions:
            new_shared_interests.update(interaction.tags)
        if new_shared_interests != self.consolidated_memory.relationship_data['shared_interests']:
            self.consolidated_memory.relationship_data['shared_interests'] = new_shared_interests
            data_updated = True

        return data_updated


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


class PersonalityManager:
    def __init__(self):
        self.default_personality_name = "default"
        # Define base paths
        self.base_path = os.path.join(os.path.dirname(__file__), 'data')
        self.personality_path = os.path.join(self.base_path, 'personalities')
        self.memory_path = os.path.join(self.base_path, 'memories')
        self.graph_path = os.path.join(self.base_path, 'graphs')
        
        # Create directories if they don't exist
        os.makedirs(self.personality_path, exist_ok=True)
        os.makedirs(self.memory_path, exist_ok=True)
        os.makedirs(self.graph_path, exist_ok=True)
    
    def get_personality_files(self, name: str) -> Dict[str, str]:
        """Get file paths for a given personality name."""
        base_name = name.lower()
        if base_name == self.default_personality_name:
            return {
                "personality": os.path.join(self.personality_path, "personality.json"),
                "short_memory": os.path.join(self.memory_path, "short_memory.json"),
                "long_memory": os.path.join(self.memory_path, "long_memory.json"),
                "consolidated_memory": os.path.join(self.memory_path, "consolidated_memory.json")
            }
        return {
            "personality": os.path.join(self.personality_path, f"{base_name}_personality.json"),
            "short_memory": os.path.join(self.memory_path, f"{base_name}_short_term_memory.json"),
            "long_memory": os.path.join(self.memory_path, f"{base_name}_long_term_memory.json"),
            "consolidated_memory": os.path.join(self.memory_path, f"{base_name}_consolidated_memory.json")
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
        """Load existing personality or create new one."""
        files = self.get_personality_files(name)
        
        # Load or create personality
        try:
            personality = PersonalityProfile.load_from_file(files["personality"])
            personality.set_name(name)  # Ensure name is set
            print(f"[Info] Loaded existing personality: {name}")
        except FileNotFoundError:
            personality = self.create_default_personality(name)
            personality.save_to_file(files["personality"])
            print(f"[Info] Created new personality: {name}")
        
        # Initialize memories
        short_memory = ShortTermMemory(max_interactions=25)
        long_memory = LongTermMemory(max_interactions=1000, personality_name=name)
        
        # Load short-term and long-term memories
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
        
        # Load consolidated memory
        memory_consolidator = MemoryConsolidator(short_memory, long_memory, personality.name)
        memory_consolidator.load_consolidated_memory(file_path=files["consolidated_memory"])
        return personality, short_memory, long_memory
    
    def save_personality_state(self, name: str, personality: PersonalityProfile, 
                            short_memory: ShortTermMemory, long_memory: LongTermMemory,
                            memory_consolidator: MemoryConsolidator):
        """Save all personality-related files."""
        files = self.get_personality_files(name)
        
        try:
            # Save personality profile
            if name != "default" or files["personality"] != "personality.json":
                personality.save_to_file(files["personality"])
            
            # Save short-term memory
            with open(files["short_memory"], "w") as f:
                json.dump(short_memory.to_list(), f, indent=4)
            
            # Save long-term memory
            with open(files["long_memory"], "w") as f:
                json.dump(long_memory.to_list(), f, indent=4)
            
            # Save consolidated memory using the MemoryConsolidator instance
            consolidated_memory_file = files["consolidated_memory"]
            memory_consolidator.save_consolidated_memory(file_path=consolidated_memory_file)
            
            # Explicitly save the memory graph
            long_memory.graph_manager.save_graph()
            
            print(f"[Debug] Saved all state files for personality: {name}")
        except Exception as e:
            print(f"[Error] Failed to save personality state: {e}")


    def list_available_personalities(self) -> List[str]:
        """List all available personalities including default"""
        personalities = set()
        
        # Debug print statements
        print(f"[Debug] Checking personality directory: {self.personality_path}")
        
        try:
            # Check for default personality
            default_path = os.path.join(self.personality_path, "personality.json")
            if os.path.exists(default_path):
                print(f"[Debug] Found default personality at {default_path}")
                personalities.add("default")
            else:
                print(f"[Debug] No default personality found at {default_path}")
            
            # Check for other personalities
            if os.path.exists(self.personality_path):
                for file in os.listdir(self.personality_path):
                    if file.endswith("_personality.json"):
                        name = file.replace("_personality.json", "")
                        print(f"[Debug] Found personality: {name}")
                        personalities.add(name)
            
            return sorted(list(personalities))
            
        except Exception as e:
            print(f"[Error] Failed to list personalities: {e}")
            return ["default"]

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
        # Pass personality name here
        self.memory_consolidator = MemoryConsolidator(short_memory, long_memory, personality.name)
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
        
        # Create interaction and generate tags first
        interaction = Interaction(user_message=query, bot_response="", tags=[])
        tags = self._generate_tags(query)
        interaction.tags = tags
        
        # Get current time
        current_time = interaction.timestamp
        
        # Get relevant context
        relevant_tags = self._extract_relevant_tags(query)
        short_term_context = self.short_memory.retrieve_relevant_interactions(query)
        long_term_context = self.long_memory.retrieve_relevant_interactions_by_tags(relevant_tags)
        
        # Add the new interaction to memories after tags are set
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
            "Time Context (BE PRECISE WITH THESE TIMES):\n"
        )
        
        # Add pattern-based context information
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
        
        # Combine and deduplicate context while preserving order
        all_context = []
        seen = set()
        for interaction in short_context + long_context:
            if interaction.timestamp not in seen:
                all_context.append(interaction)
                seen.add(interaction.timestamp)
        
        # Sort by timestamp
        all_context.sort(key=lambda x: x.timestamp)
        
        # Format context with clearer temporal markers
        context_strings = []
        for inter in all_context:
            time_ago = (datetime.now() - inter.timestamp).total_seconds()
            time_str = self._format_time(time_ago)
            context_strings.append(
                f"[{time_str} ago]\n"
                f"User: {inter.user_message}\n"
                f"Bot: {inter.bot_response}"
            )
        
        if context_strings:
            system_content += f"\nContext:\n{chr(10).join(context_strings)}\n"
        
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
            max_tokens=2000,
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
                    {"role": "system", "content": "Extract relevant topics from the text as a comma-separated list. Include both specific topics and general categories. Do not include numbers or explanations."},
                    {"role": "user", "content": query}
                ],
                max_tokens=60,
                temperature=0.3,
            )
            # Clean and process the response
            tags_text = response.choices[0].message.content.strip()
            # Split by comma and clean each tag
            tags = [tag.strip().lower() for tag in tags_text.split(',') 
                if tag.strip() and not tag.startswith(('1.', '2.', '3.'))]
            print(f"[Debug] Extracted tags: {tags}")
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
        chatbot.memory_consolidator  # Pass the MemoryConsolidator instance here
    )
    print(f"Session saved for personality: {personality_name}. Goodbye!")

if __name__ == "__main__":
    main()