import json
import networkx as nx
from datetime import datetime
import numpy as np
from collections import defaultdict

def load_memories(file_path):
    """Load memories from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_tag_similarity(tags1, tags2):
    """Calculate Jaccard similarity between two sets of tags."""
    if not tags1 or not tags2:
        return 0
    set1 = set(tags1)
    set2 = set(tags2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def calculate_temporal_proximity(time1, time2, max_time_diff=86400):  # 24 hours in seconds
    """Calculate temporal proximity between two timestamps."""
    t1 = datetime.fromisoformat(time1)
    t2 = datetime.fromisoformat(time2)
    time_diff = abs((t1 - t2).total_seconds())
    return max(0, 1 - (time_diff / max_time_diff))

def create_memory_graph(memories):
    """Create a NetworkX graph from memories."""
    G = nx.Graph()
    
    # Create nodes
    for idx, memory in enumerate(memories):
        node_id = idx
        G.add_node(node_id, 
                   message=memory['user_message'],
                   response=memory['bot_response'],
                   timestamp=memory['timestamp'],
                   tags=memory['tags'],
                   priority=memory.get('priority_score', 0.5))
    
    # Create edges based on tag similarity and temporal proximity
    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            # Calculate tag similarity
            tag_sim = calculate_tag_similarity(memories[i]['tags'], memories[j]['tags'])
            
            # Calculate temporal proximity
            temp_prox = calculate_temporal_proximity(memories[i]['timestamp'], 
                                                   memories[j]['timestamp'])
            
            # Combine scores with weights
            edge_weight = (0.7 * tag_sim) + (0.3 * temp_prox)
            
            # Add edge if weight is significant
            if edge_weight > 0.1:  # Threshold to avoid too many weak connections
                G.add_edge(i, j, weight=edge_weight)
    
    return G

def analyze_graph(G):
    """Analyze the graph structure."""
    analysis = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'communities': len(list(nx.community.greedy_modularity_communities(G))),
    }
    return analysis

def find_similar_memories(G, query_tags, top_n=5):
    """Find most similar memories to a query based on tags."""
    similarities = []
    
    # Calculate similarity for each node
    for node in G.nodes():
        node_tags = G.nodes[node]['tags']
        sim = calculate_tag_similarity(query_tags, node_tags)
        similarities.append((node, sim))
    
    # Sort by similarity and return top N
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

def save_graph(G, file_path):
    """Save graph to JSON format."""
    graph_data = {
        'nodes': [[n, data] for n, data in G.nodes(data=True)],
        'edges': [[u, v, data] for u, v, data in G.edges(data=True)]
    }
    
    with open(file_path, 'w') as f:
        json.dump(graph_data, f, indent=4)

def main():
    # Load memories
    memories = load_memories('long_memory.json')
    
    # Create graph
    G = create_memory_graph(memories)
    
    # Analyze graph
    analysis = analyze_graph(G)
    print("\nGraph Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    # Save graph
    save_graph(G, 'memory_graph.json')
    print("\nGraph saved to memory_graph.json")
    
    # Example of finding similar memories
    example_query = ["coding", "memory", "improvement"]
    similar_memories = find_similar_memories(G, example_query)
    print("\nExample Query Results:")
    for node_id, similarity in similar_memories:
        print(f"Memory {node_id}: Similarity {similarity:.2f}")
        print(f"Message: {G.nodes[node_id]['message'][:100]}...")

if __name__ == "__main__":
    main()