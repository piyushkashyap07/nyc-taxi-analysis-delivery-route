"""
Route Optimization Module
Handles graph-based route optimization and network analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import folium

def route_optimization_analysis(df, sample_size=20):
    """
    Perform graph-based route optimization analysis
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        sample_size (int): Number of locations to include in network
        
    Returns:
        tuple: (G, route_stats) - Network graph and route statistics
    """
    print("\n" + "="*50)
    print("4. GRAPH & ROUTE OPTIMIZATION")
    print("="*50)
    
    if 'PULocationID' not in df.columns or 'DOLocationID' not in df.columns:
        print("Location columns not found. Skipping route optimization.")
        return None, None
    
    print(f"\n4.1 Network Construction (Sample: {sample_size} locations)")
    print("-" * 50)
    
    # Select top pickup locations
    top_locations = df['PULocationID'].value_counts().head(sample_size).index.tolist()
    
    # Create subset with these locations
    df_subset = df[df['PULocationID'].isin(top_locations) & 
                   df['DOLocationID'].isin(top_locations)].copy()
    
    print(f"Selected locations: {top_locations[:10]}...")
    print(f"Network subset size: {len(df_subset)} trips")
    
    # Calculate average travel times between locations
    route_stats = df_subset.groupby(['PULocationID', 'DOLocationID']).agg({
        'trip_duration_minutes': ['mean', 'count'],
        'trip_distance': 'mean'
    }).round(2)
    
    route_stats.columns = ['avg_duration', 'trip_count', 'avg_distance']
    route_stats = route_stats.reset_index()
    route_stats = route_stats[route_stats['trip_count'] >= 5]  # Filter for reliability
    
    print(f"Valid routes in network: {len(route_stats)}")
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_nodes_from(top_locations)
    
    # Add edges with weights (travel time)
    for _, row in route_stats.iterrows():
        G.add_edge(row['PULocationID'], row['DOLocationID'], 
                  weight=row['avg_duration'],
                  distance=row['avg_distance'],
                  trips=row['trip_count'])
    
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 4.2 Route Optimization
    print("\n4.2 Route Optimization Algorithms")
    print("-" * 35)
    
    # Shortest path example
    if len(top_locations) >= 2:
        source = top_locations[0]
        target = top_locations[1]
        
        try:
            shortest_path = nx.shortest_path(G, source, target, weight='weight')
            path_length = nx.shortest_path_length(G, source, target, weight='weight')
            
            print(f"Shortest path from {source} to {target}:")
            print(f"Route: {' -> '.join(map(str, shortest_path))}")
            print(f"Total time: {path_length:.1f} minutes")
        except nx.NetworkXNoPath:
            print(f"No path found between {source} and {target}")
    
    # TSP approximation for delivery route
    print("\n4.3 Delivery Route Optimization (TSP Approximation)")
    print("-" * 50)
    
    # Select subset for TSP (computational efficiency)
    tsp_locations = top_locations[:min(8, len(top_locations))]
    
    if len(tsp_locations) >= 3:
        # Create complete graph for TSP
        G_complete = nx.Graph()
        
        # Add all combinations of locations
        for loc1, loc2 in combinations(tsp_locations, 2):
            # Find shortest path in original graph or use average
            try:
                if G.has_edge(loc1, loc2):
                    weight = G[loc1][loc2]['weight']
                elif G.has_edge(loc2, loc1):
                    weight = G[loc2][loc1]['weight']
                else:
                    # Use average duration for missing edges
                    weight = df['trip_duration_minutes'].mean()
                
                G_complete.add_edge(loc1, loc2, weight=weight)
            except:
                weight = df['trip_duration_minutes'].mean()
                G_complete.add_edge(loc1, loc2, weight=weight)
        
        # Approximate TSP solution using nearest neighbor
        optimal_route, total_time = nearest_neighbor_tsp(G_complete, tsp_locations[0])
        
        print(f"Optimized delivery route:")
        print(f"Route: {' -> '.join(map(str, optimal_route))}")
        print(f"Total time: {total_time:.1f} minutes")
        print(f"Average time per stop: {total_time/len(tsp_locations):.1f} minutes")
    
    # 4.4 Network Visualization
    print("\n4.4 Network Visualization")
    print("-" * 25)
    
    plt.figure(figsize=(12, 8))
    
    # Position nodes
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=300, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, 
                          edge_color='gray', arrowsize=10)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('NYC Taxi Route Network\n(Top Pickup Locations)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/network_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4.5 Business Applications
    print("\n4.5 Business Applications & Scalability")
    print("-" * 42)
    
    applications = [
        "• Delivery Route Optimization: Minimize total travel time for package deliveries",
        "• Ride-sharing Fleet Management: Optimize driver positioning and routes",
        "• Dynamic Pricing: Use network congestion data for surge pricing",
        "• Demand Forecasting: Predict traffic patterns between location pairs",
        "• Infrastructure Planning: Identify critical routes needing capacity improvements"
    ]
    
    scalability_notes = [
        "• Graph Partitioning: Divide city into clusters for computational efficiency",
        "• Real-time Updates: Integrate live traffic data for dynamic route adjustment",
        "• Machine Learning: Use historical patterns to predict optimal routes",
        "• Cloud Computing: Leverage distributed systems for large-scale optimization",
        "• API Integration: Connect with mapping services for real-world implementation"
    ]
    
    print("Business Applications:")
    for app in applications:
        print(app)
    
    print("\nScalability Considerations:")
    for note in scalability_notes:
        print(note)
    
    return G, route_stats

def nearest_neighbor_tsp(graph, start_node):
    """
    Approximate TSP solution using nearest neighbor algorithm
    
    Args:
        graph (nx.Graph): Complete graph with edge weights
        start_node: Starting node for the route
        
    Returns:
        tuple: (path, total_weight) - Optimal route and total time
    """
    unvisited = set(graph.nodes()) - {start_node}
    current = start_node
    path = [current]
    total_weight = 0
    
    while unvisited:
        nearest = min(unvisited, 
                     key=lambda x: graph[current][x]['weight'] if graph.has_edge(current, x) else float('inf'))
        total_weight += graph[current][nearest]['weight']
        current = nearest
        path.append(current)
        unvisited.remove(current)
    
    # Return to start
    if graph.has_edge(current, start_node):
        total_weight += graph[current][start_node]['weight']
        path.append(start_node)
    
    return path, total_weight

def geospatial_analysis(df):
    """
    Perform geospatial analysis if coordinates are available
    
    Args:
        df (pd.DataFrame): Cleaned dataset
    """
    coord_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    
    if not any(col in df.columns for col in coord_cols):
        print("\nGeospatial coordinates not available in dataset.")
        return
    
    print("\n" + "="*50)
    print("GEOSPATIAL ANALYSIS")
    print("="*50)
    
    # Filter valid coordinates
    valid_coords = df.dropna(subset=[col for col in coord_cols if col in df.columns])
    
    if len(valid_coords) == 0:
        print("No valid coordinate data found.")
        return
    
    print(f"Trips with valid coordinates: {len(valid_coords)}")
    
    # Basic coordinate statistics
    for col in coord_cols:
        if col in valid_coords.columns:
            print(f"{col}: {valid_coords[col].min():.4f} to {valid_coords[col].max():.4f}")
    
    # Create a sample map (requires folium)
    try:
        # Sample data for mapping
        sample_trips = valid_coords.sample(n=min(1000, len(valid_coords)))
        
        # Center map on NYC
        center_lat = sample_trips['pickup_latitude'].mean()
        center_lon = sample_trips['pickup_longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Add pickup points
        for idx, row in sample_trips.head(100).iterrows():
            folium.CircleMarker(
                location=[row['pickup_latitude'], row['pickup_longitude']],
                radius=2,
                color='blue',
                fillColor='blue',
                popup=f"Trip Duration: {row.get('trip_duration_minutes', 'N/A'):.1f} min"
            ).add_to(m)
        
        # Save map
        m.save('outputs/nyc_taxi_pickup_map.html')
        print("Interactive map saved as 'outputs/nyc_taxi_pickup_map.html'")
        
    except ImportError:
        print("Folium not available for mapping. Install with: pip install folium")

def calculate_route_efficiency(df, route_locations):
    """
    Calculate efficiency metrics for a given route
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        route_locations (list): List of location IDs in route order
        
    Returns:
        dict: Efficiency metrics
    """
    if len(route_locations) < 2:
        return None
    
    total_time = 0
    total_distance = 0
    route_segments = []
    
    for i in range(len(route_locations) - 1):
        start_loc = route_locations[i]
        end_loc = route_locations[i + 1]
        
        # Find trips between these locations
        segment_trips = df[(df['PULocationID'] == start_loc) & 
                          (df['DOLocationID'] == end_loc)]
        
        if len(segment_trips) > 0:
            avg_time = segment_trips['trip_duration_minutes'].mean()
            avg_distance = segment_trips['trip_distance'].mean()
            trip_count = len(segment_trips)
            
            total_time += avg_time
            total_distance += avg_distance
            
            route_segments.append({
                'from': start_loc,
                'to': end_loc,
                'avg_time': avg_time,
                'avg_distance': avg_distance,
                'trip_count': trip_count
            })
    
    efficiency_metrics = {
        'total_time': total_time,
        'total_distance': total_distance,
        'segments': route_segments,
        'avg_time_per_segment': total_time / len(route_segments) if route_segments else 0,
        'total_segments': len(route_segments)
    }
    
    return efficiency_metrics 