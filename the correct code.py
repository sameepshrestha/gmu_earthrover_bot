the correct code  this is not .py file 


38.8164176940918	-77.2771759033203

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import numpy as np
from pyproj import Proj
import pandas as pd
import os

# Function definitions (unchanged unless noted)
def get_utm_graph_from_point(center_point, dist=1000):
    G = ox.graph_from_point(center_point, dist=dist, network_type='drive')
    G_utm = ox.project_graph(G)
    edges = ox.graph_to_gdfs(G_utm, nodes=False)
    return G_utm, edges

def get_utm_graph_from_gps(gps_start, gps_goal):
    lat1, lon1 = gps_start
    lat2, lon2 = gps_goal
    center_lat = (lat1 + lat2) / 2
    center_lon = (lon1 + lon2) / 2
    center_point = (center_lat, center_lon)
    dist = ox.distance.great_circle(lat1, lon1, lat2, lon2) + 500
    G_utm, edges = get_utm_graph_from_point(center_point, dist)
    utm_zone = int((center_lon + 180) / 6) + 1
    G_utm.graph['utm_zone'] = utm_zone
    return G_utm, edges

def latlon_to_utm(lat, lon, utm_zone=None):
    if utm_zone is None:
        utm_zone = int((lon + 180) / 6) + 1
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    easting, northing = utm_proj(lon, lat)
    return easting, northing, utm_zone

def utm_to_latlon(easting, northing, utm_zone):
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    lon, lat = utm_proj(easting, northing, inverse=True)
    return lat, lon

def calculate_bearing(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2
    delta_lon = lon2 - lon1
    x = np.cos(np.radians(lat2)) * np.sin(np.radians(delta_lon))
    y = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
        np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(delta_lon))
    initial_bearing = np.degrees(np.arctan2(x, y))
    return (initial_bearing + 360) % 360

def determine_turn_direction(prev_point, current_point, next_point):
    bearing1 = calculate_bearing(prev_point, current_point)
    bearing2 = calculate_bearing(current_point, next_point)
    turn_angle = (bearing2 - bearing1 + 360) % 360
    if 20 < turn_angle < 170:
        return "RIGHT"
    elif 200 < turn_angle < 350:
        return "LEFT"
    return "STRAIGHT"

def snap_gps_to_edge(G: nx.MultiDiGraph, gps_point, edges):
    lat, lon = gps_point
    easting, northing, utm_zone = latlon_to_utm(lat, lon)
    gps_utm = Point(easting, northing)
    edge = ox.distance.nearest_edges(G, X=easting, Y=northing)
    edge_geom = edges.loc[edge, 'geometry']
    snapped_point = edge_geom.interpolate(edge_geom.project(gps_utm))
    return snapped_point, edge

def add_temporary_nodes(G, snapped_point, edge):
    new_node_id = max(G.nodes) + 1
    G.add_node(new_node_id, x=snapped_point.x, y=snapped_point.y)
    u, v, key = edge
    dist_to_v = np.sqrt((snapped_point.x - G.nodes[v]['x'])**2 + (snapped_point.y - G.nodes[v]['y'])**2)
    dist_to_u = np.sqrt((snapped_point.x - G.nodes[u]['x'])**2 + (snapped_point.y - G.nodes[u]['y'])**2)
    G.add_edge(new_node_id, v, length=dist_to_v)
    G.add_edge(u, new_node_id, length=dist_to_u)
    return new_node_id

def compute_path(G: nx.MultiDiGraph, start_node: int, goal_node: int):
    route = nx.astar_path(
        G, start_node, goal_node,
        weight='length',
        heuristic=lambda n1, n2: np.sqrt(
            (G.nodes[n1]['x'] - G.nodes[n2]['x'])**2 + 
            (G.nodes[n1]['y'] - G.nodes[n2]['y'])**2
        )
    )
    return route

def track_progress(G: nx.MultiDiGraph, prev_node, current_gps, next_node):
    lat, lon = current_gps
    easting, northing, _ = latlon_to_utm(lat, lon, utm_zone=G.graph['utm_zone'])
    current_utm = Point(easting, northing)
    prev_x, prev_y = G.nodes[prev_node]['x'], G.nodes[prev_node]['y']
    next_x, next_y = G.nodes[next_node]['x'], G.nodes[next_node]['y']
    edge_data = G.edges[(prev_node, next_node, 0)]
    edge_geom = edge_data.get('geometry', LineString([(prev_x, prev_y), (next_x, next_y)]))
    proj_distance = edge_geom.project(current_utm)
    projected_point = edge_geom.interpolate(proj_distance)
    d_start = np.sqrt((prev_x - projected_point.x)**2 + (prev_y - projected_point.y)**2)
    d_end = np.sqrt((projected_point.x - next_x)**2 + (projected_point.y - next_y)**2)
    total = np.sqrt((prev_x - next_x)**2 + (prev_y - next_y)**2)
    prev_latlon = utm_to_latlon(prev_x, prev_y, G.graph['utm_zone'])
    next_latlon = utm_to_latlon(next_x, next_y, G.graph['utm_zone'])
    return d_start, d_end, total, projected_point, prev_latlon, next_latlon

def get_route_turns(G, route):
    turns = []
    for i in range(1, len(route) - 1):
        prev_latlon = utm_to_latlon(G.nodes[route[i-1]]['x'], G.nodes[route[i-1]]['y'], G.graph['utm_zone'])
        current_latlon = utm_to_latlon(G.nodes[route[i]]['x'], G.nodes[route[i]]['y'], G.graph['utm_zone'])
        next_latlon = utm_to_latlon(G.nodes[route[i+1]]['x'], G.nodes[route[i+1]]['y'], G.graph['utm_zone'])
        turn = determine_turn_direction(prev_latlon, current_latlon, next_latlon)
        turns.append(turn)
    return turns

def plot_map(G, current_utm, prev_coords, next_coords, projected_point, route_coords, save_path, frame, turn):
    fig, ax = ox.plot_graph(G, node_size=0, show=False, close=False)
    # Plot route
    route_x, route_y = zip(*route_coords)
    ax.plot(route_x, route_y, c='yellow', linewidth=2, label="Route")
    # Plot points in UTM
    ax.scatter(prev_coords[0], prev_coords[1], c='red', marker='o', s=50, label="Previous Node")
    ax.scatter(next_coords[0], next_coords[1], c='red', marker='o', s=50, label="Next Node")
    ax.scatter(current_utm.x, current_utm.y, c='green', marker='x', s=100, label="Current GPS")
    ax.scatter(projected_point.x, projected_point.y, c='blue', marker='s', s=50, label="Projected Point")
    ax.legend()
    ax.set_title(f"Frame {frame:03d} - Tracking Progress")
    if turn:
        ax.text(current_utm.x, current_utm.y + 10, f"Turn: {turn}", fontsize=12, color='blue', ha='center')
    plt.savefig(f"{save_path}/frame_{frame:03d}.png")
    plt.close(fig)

# Main execution
save_path = "/home/sameep/earth-rovers-sdk/images_folder"
os.makedirs(save_path, exist_ok=True)
data = pd.read_csv("/home/sameep/Downloads/main_data_collection.csv")
print(f"Data shape: {data.shape}")
gps_goal = (data["latitude"].iloc[-1], data["longitude"].iloc[-1])  # Last point as goal
gps_start = (data["latitude"].iloc[0], data["longitude"].iloc[0])    # First point as start

# Load graph
G_utm, edges = get_utm_graph_from_gps(gps_start, gps_goal)
snapped_start, edge_start = snap_gps_to_edge(G_utm, gps_start, edges)
snapped_goal, edge_goal = snap_gps_to_edge(G_utm, gps_goal, edges)
start_node = add_temporary_nodes(G_utm, snapped_start, edge_start)
goal_node = add_temporary_nodes(G_utm, snapped_goal, edge_goal)

route = compute_path(G_utm, start_node, goal_node)
turns = get_route_turns(G_utm, route)
route_coords = [(G_utm.nodes[node]['x'], G_utm.nodes[node]['y']) for node in route]

# Simulate GPS movement
distance_threshold = 4  # meters
node_idx = 0  # Start at first route node

for i in range(len(data)):
    current_gps = (data["latitude"].iloc[i], data["longitude"].iloc[i])
    if i > 0 and current_gps == (data["latitude"].iloc[i-1], data["longitude"].iloc[i-1]):
        continue
    
    # Convert current GPS to UTM
    easting, northing, _ = latlon_to_utm(current_gps[0], current_gps[1], G_utm.graph['utm_zone'])
    current_utm = Point(easting, northing)
    
    # Determine previous and next nodes
    prev_node = route[node_idx]
    next_node = route[min(node_idx + 1, len(route) - 1)]
    d_start, d_end, total, projected_point, prev_latlon, next_latlon = track_progress(G_utm, prev_node, current_gps, next_node)
    
    # Update node index if close to next node
    if d_end <= distance_threshold and node_idx < len(route) - 2:
        node_idx += 1
        prev_node = route[node_idx]
        next_node = route[min(node_idx + 1, len(route) - 1)]
        d_start, d_end, total, projected_point, prev_latlon, next_latlon = track_progress(G_utm, prev_node, current_gps, next_node)
    
    # Get turn direction
    turn = turns[node_idx - 1] if node_idx > 0 and node_idx - 1 < len(turns) else None
    
    # Plot with UTM coordinates
    prev_coords = (G_utm.nodes[prev_node]['x'], G_utm.nodes[prev_node]['y'])
    next_coords = (G_utm.nodes[next_node]['x'], G_utm.nodes[next_node]['y'])
    plot_map(G_utm, current_utm, prev_coords, next_coords, projected_point, route_coords, save_path, i, turn)
    
    if i > 2000:
        print(f"Frame {i}: d_end={d_end:.2f}m")
it works but the print has not been workingfor me the new code and plot map 