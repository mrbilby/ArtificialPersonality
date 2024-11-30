import json
from datetime import datetime

def clean_memory_graph(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    nodes = data.get('nodes', [])
    edges = data.get('edges', [])

    cleaned_nodes = []
    node_id_map = {}
    new_id = 0

    for node_id, node_data in nodes:
        # Check for missing 'tags' or 'timestamp'
        if 'tags' not in node_data or 'timestamp' not in node_data:
            print(f"Removing node {node_id} due to missing 'tags' or 'timestamp'.")
            continue  # Skip this node

        # Validate 'timestamp'
        timestamp_str = node_data['timestamp']
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            if timestamp > datetime.now():
                print(f"Adjusting future timestamp for node {node_id}.")
                node_data['timestamp'] = datetime.now().isoformat()
        except ValueError:
            print(f"Invalid timestamp format for node {node_id}. Setting to current time.")
            node_data['timestamp'] = datetime.now().isoformat()

        # Assign a new sequential node ID
        node_id_map[node_id] = new_id
        cleaned_nodes.append([new_id, node_data])
        new_id += 1

    # Rebuild edges with new node IDs, and remove edges connected to removed nodes
    cleaned_edges = []
    for u, v, edge_data in edges:
        if u in node_id_map and v in node_id_map:
            cleaned_edges.append([node_id_map[u], node_id_map[v], edge_data])
        else:
            print(f"Removing edge ({u}, {v}) due to missing node.")

    # Write cleaned data back to file
    cleaned_data = {'nodes': cleaned_nodes, 'edges': cleaned_edges}
    with open(file_path, 'w') as f:
        json.dump(cleaned_data, f, indent=4)
    print("Memory graph cleaned successfully.")

# Use the function
clean_memory_graph('carl_memory_graph.json')
