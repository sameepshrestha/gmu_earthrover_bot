import time
import requests
import geopandas as gpd
import osmnx as ox
import numpy as np
import networkx as nx
import pandas as pd
from shapely.geometry import Point
from botreceiver import BotReceiver
import math
import matplotlib.pyplot as plt
import cv2 
from io import BytesIO

class GPSRouteTracker:
    def __init__(self, start_gps, destination_gps):
        self.start_gps = start_gps
        self.destination_gps = destination_gps
        self.G = ox.graph_from_point(start_gps, dist=500, dist_type="network", network_type="walk", simplify=True)
        self.graph_proj = ox.project_graph(self.G)
        self.nodes, self.edges = ox.graph_to_gdfs(self.graph_proj)
        self.crs_proj = self.graph_proj.graph['crs']
        # self.data = pd.read_csv(data_file)
        #addition for the plot, storing a stationary plot 
        self.origin = gpd.GeoDataFrame({'name': ['Origin'], 'geometry': [Point(start_gps[1], start_gps[0])]}, crs="EPSG:4326").to_crs(crs=self.crs_proj)
        self.destination = gpd.GeoDataFrame({'name': ['Destination'], 'geometry': [Point(destination_gps[1], destination_gps[0])]}, crs="EPSG:4326").to_crs(crs=self.crs_proj)

        self.origin_node = ox.distance.nearest_nodes(self.graph_proj, X=float(self.origin.geometry.x), Y=float(self.origin.geometry.y))
        self.destination_node = ox.distance.nearest_nodes(self.graph_proj, X=float(self.destination.geometry.x), Y=float(self.destination.geometry.y))
        
        self.route = nx.shortest_path(self.graph_proj, self.origin_node, self.destination_node, weight='length')
        self.route_utm_coords = [(self.nodes.loc[node].geometry.x, self.nodes.loc[node].geometry.y) for node in self.route]
        self.route_gps_coords = [self.utm_to_gps(x, y) for x, y in self.route_utm_coords]
        self.adjust_route_with_edge_snapping()
        self.check_proximity()
        # this is again for trhe plot , just defining the range of the plot and saving a temporary file 
        self.mapfile = "/home/kintou/Work/Robotixx/Sameep_PC/frodobot_final_second/frodobot_final/robot_navigation/output_folder/frame_001.png"
        utm_xs, utm_ys = zip(*self.route_utm_coords)
        self.utm_min_x, self.utm_max_x = min(utm_xs), max(utm_xs)
        self.utm_min_y, self.utm_max_y = min(utm_ys), max(utm_ys)
        padding = 50
        self.utm_min_x -= padding
        self.utm_max_x += padding
        self.utm_min_y -= padding
        self.utm_max_y += padding
        self.fig, self.ax = self._draw_static_map()

    def gps_to_utm(self, latitude, longitude):
        gps_point = gpd.GeoDataFrame({'geometry': [Point(longitude, latitude)]}, crs="EPSG:4326").to_crs(crs=self.crs_proj)
        return gps_point.geometry.x.iloc[0], gps_point.geometry.y.iloc[0]

    def utm_to_gps(self, utm_x, utm_y):
        utm_point = gpd.GeoDataFrame({'geometry': [Point(utm_x, utm_y)]}, crs=self.crs_proj).to_crs(crs="EPSG:4326")
        return utm_point.geometry.y.iloc[0], utm_point.geometry.x.iloc[0]
    def get_route(self):
        return self.route_gps_coords
    
    def get_distance_to_next(self, current_gps, next_node_index=0):
        utm_x, utm_y = self.gps_to_utm(current_gps[0], current_gps[1])
        current_utm = Point(utm_x, utm_y)
        next_utm = Point(self.route_utm_coords[next_node_index][0], self.route_utm_coords[next_node_index][1])
        dx = next_utm.x - current_utm.x  # Difference in easting
        dy = next_utm.y - current_utm.y  # Difference in northing
        bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        return current_utm.distance(Point(self.route_utm_coords[next_node_index][0], self.route_utm_coords[next_node_index][1])), bearing
    
    def check_proximity(self, threshold=5):

        # if len(self.route_gps_coords) == 1: 
        origin_utm_x, original_utm_y= self.gps_to_utm(self.start_gps[0],self.start_gps[1])
        destination_utm_x, destination_utm_y = self.gps_to_utm(self.destination_gps[0],self.destination_gps[0])
        origin_utm = Point(origin_utm_x,original_utm_y)
        destination_utm = Point(destination_utm_x,destination_utm_y)
        if origin_utm.distance(Point(self.route_utm_coords[0][0], self.route_utm_coords[0][1])) >threshold:
          self.route_utm_coords.insert(0, (origin_utm_x, original_utm_y))
          self.route_gps_coords.insert(0, (self.start_gps[0], self.start_gps[1]))
        if destination_utm.distance(Point(self.route_utm_coords[-1][0], self.route_utm_coords[-1][1])) >threshold:
          self.route_utm_coords.append((destination_utm_x, destination_utm_y))
          self.route_gps_coords.append((self.destination_gps[0], self.destination_gps[1]))


    def adjust_route_with_edge_snapping(self):
        current_utm= self.gps_to_utm(self.start_gps[0],self.start_gps[1])
        current_utm = Point(current_utm[0], current_utm[1])
        u, v, key = ox.distance.nearest_edges(self.graph_proj, current_utm.x, current_utm.y)
        count = 0 
        if u in self.route and v in self.route:
            count+=2
        elif u in self.route or v in self.route:
            count+=1
        # else :
        #     ("path expected to be be wrong adjusting the closest node ")
        #     route1 = nx.shortest_path(self.graph_proj, u, self.destination_node, weight='length')
        #     route2 = nx.shortest_path(self.graph_proj, v, self.destination_node, weight = 'length')
        #             # Compute their costs using shortest_path_length
        #     cost1 = nx.shortest_path_length(self.graph_proj, u, self.destination_node, weight='length')
        #     cost2 = nx.shortest_path_length(self.graph_proj, v, self.destination_node, weight='length')
        #     if cost1 <= cost2:
        #         chosen_route = route1
        #     else:
        #         chosen_route = route2
        #     self.route = chosen_route
        #     self.route_utm_coords = [(self.nodes.loc[node].geometry.x, self.nodes.loc[node].geometry.y) for node in chosen_route]
        #     self.route_gps_coords = [self.utm_to_gps(x, y) for x, y in self.route_utm_coords]
            # If the snapped edge isn’t part of your route,
            # you might want to re-run the route calculation from the current position.
        if count ==2:
            self.route_utm_coords.pop(0)  
            self.route_gps_coords.pop(0)
        else:
            print("False, no changes ")

    def _draw_static_map(self):
            """Draws the static GPS map and returns the figure and axes."""
            fig, ax = ox.plot_graph(self.graph_proj, node_size=10, edge_color='gray', edge_linewidth=1, show=False, close=False)
            route_nodes = self.nodes.loc[self.route]
            ax.plot(route_nodes.geometry.x, route_nodes.geometry.y, color='red', linewidth=2, marker='o', markersize=5)
            ax.scatter([self.origin.geometry.x.iloc[0]], [self.origin.geometry.y.iloc[0]], c='blue', s=100, label='Start')
            ax.scatter([self.destination.geometry.x.iloc[0]], [self.destination.geometry.y.iloc[0]], c='green', s=100, label='Goal')
            plt.axis("off")
            return fig, ax
    
    def plot_map(self, current_gps):
            """Overlay current GPS on the cached figure and return as an image."""
            # Work with a fresh copy of the figure to avoid accumulating GPS points
            fig = self.fig  # We’ll clone this if needed, but for now assume one-time use
            ax = self.ax
            
            # Convert current GPS to UTM and plot it
            current_x, current_y = self.gps_to_utm(current_gps[0], current_gps[1])
            ax.scatter([current_x], [current_y], c='yellow', s=100, label='Current', zorder=10)
            
            # Render the figure to a NumPy array
            fig.canvas.draw()
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            buf.close()
            
            # Resize to target size
            
            return img