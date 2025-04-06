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
from matplotlib.animation import FuncAnimation
#TODO: Include yaml

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
        #TODO: Include the CRS in the global YAML File
        self.destination = gpd.GeoDataFrame({'name': ['Destination'], 'geometry': [Point(destination_gps[1], destination_gps[0])]}, crs="EPSG:4326").to_crs(crs=self.crs_proj)

        #Get the connection with the nearest node.
        self.origin_node = ox.distance.nearest_nodes(self.graph_proj, X=float(self.origin.geometry.x), Y=float(self.origin.geometry.y))
        self.destination_node = ox.distance.nearest_nodes(self.graph_proj, X=float(self.destination.geometry.x), Y=float(self.destination.geometry.y))

        #Get the 1st shortest path but you can get n shortest paths     
        self.route = nx.shortest_path(self.graph_proj, self.origin_node, self.destination_node, weight='length')
        self.route_length = len(self.route)
        self.route_utm_coords = [(self.nodes.loc[node].geometry.x, self.nodes.loc[node].geometry.y) for node in self.route]
        self.route_gps_coords = [self.utm_to_gps(x, y) for x, y in self.route_utm_coords]
        # Check where the robot is and if needed find the connection to the nearest edge
        self.adjust_route_with_edge_snapping()      
        self.check_proximity()              #check if the robot  is near tthe next node and adjust the route accordingly
        
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
        # self.fig, self.ax = self._draw_static_map()

    def gps_to_utm(self, latitude, longitude):
        gps_point = gpd.GeoDataFrame({'geometry': [Point(longitude, latitude)]}, crs="EPSG:4326").to_crs(crs=self.crs_proj)
        return gps_point.geometry.x.iloc[0], gps_point.geometry.y.iloc[0]

    def utm_to_gps(self, utm_x, utm_y):
        utm_point = gpd.GeoDataFrame({'geometry': [Point(utm_x, utm_y)]}, crs=self.crs_proj).to_crs(crs="EPSG:4326")
        #TODO, Moving CRS to yaml
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
    
    #for utm 
    def get_distance_to_next_utm(self, current_utm, start_utm, next_node_index=0):
        utm_x, utm_y = (current_utm[0], current_utm[1])
        current_utm = Point(utm_x, utm_y)
        next_utm = Point(self.route_utm_coords[next_node_index][0], self.route_utm_coords[next_node_index][1])
        print(utm_x, utm_y, "this is the current utm", next_utm)

        dx = next_utm.x - current_utm.x  # Difference in easting
        dy = next_utm.y - current_utm.y  # Difference in northing
        bearing = math.atan2(dy,dx) 
        bearing = (bearing + 2 * math.pi) % (2 * math.pi)
        projected_distance = self.get_projected_distance(current_utm, next_utm)
        return current_utm.distance(Point(self.route_utm_coords[next_node_index][0], self.route_utm_coords[next_node_index][1])), projected_distance, bearing

    def get_projected_distance(self, current_gps, next_node_index=0):
        current_utm = self.gps_to_utm(current_gps[0], current_gps[1])
        current_utm = Point(current_utm[0], current_utm[1])

        next_utm = Point(self.route_utm_coords[next_node_index][0], self.route_utm_coords[next_node_index][1])
        u, v, key = ox.distance.nearest_edges(self.graph_proj, current_utm.x, current_utm.y)
        edge_geom = self.edges.loc[(u, v, key), 'geometry'] 
        if edge_geom.geom_type == 'LineString':  # Ensure it's a valid edge
            projected_point = edge_geom.interpolate(edge_geom.project(current_utm))
            projected_distance = next_utm.distance(projected_point)  # Perpendicular distance
        else:
            projected_distance = None
        return projected_distance
    
    def get_projected_distance_copy(self, current_gps, next_node_index=0):
        current_utm = self.gps_to_utm(current_gps[0], current_gps[1])
        current_utm = Point(current_utm[0], current_utm[1])

        next_utm = Point(self.route_utm_coords[next_node_index][0], self.route_utm_coords[next_node_index][1])
        u, v, key = ox.distance.nearest_edges(self.graph_proj, current_utm.x, current_utm.y)
        edge_geom = self.edges.loc[(u, v, key), 'geometry'] 
        if edge_geom.geom_type == 'LineString':  # Ensure it's a valid edge
            projected_point = edge_geom.interpolate(edge_geom.project(current_utm))
            projected_distance = next_utm.distance(projected_point)  # Perpendicular distance
        else:
            projected_distance = None
        return projected_distance


    def check_proximity(self, threshold=5):

        # if len(self.route_gps_coords) == 1: 
        origin_utm_x, original_utm_y= self.gps_to_utm(self.start_gps[0],self.start_gps[1])
        destination_utm_x, destination_utm_y = self.gps_to_utm(self.destination_gps[0],self.destination_gps[1])
        origin_utm = Point(origin_utm_x,original_utm_y)
        destination_utm = Point(destination_utm_x,destination_utm_y)
        if origin_utm.distance(Point(self.route_utm_coords[0][0], self.route_utm_coords[0][1])) >threshold:
          self.route_utm_coords.insert(0, (origin_utm_x, original_utm_y))
          self.route_gps_coords.insert(0, (self.start_gps[0], self.start_gps[1]))
        if destination_utm.distance(Point(self.route_utm_coords[-1][0], self.route_utm_coords[-1][1])) >threshold:
          self.route_utm_coords.append((destination_utm_x, destination_utm_y))
          self.route_gps_coords.append((self.destination_gps[0], self.destination_gps[1]))

    # def recalibrate(self, botcontroller, ros_publisher, node, duration=10, threshold = 20):
    #     start_time = time.time()
    #     steering = 0.1
        
    #     while time.time() - start_time < duration:
    #         current_utm, current_orientation =ros_publisher.get_latest_data()

    #         if current_utm != None:
    #             _, _, bearing = self.get_distance_to_next_utm(current_utm, current_utm, node) 
    #         else:

    #         delta_angle = (bearing - current_orientation + 360) % 360
    #         if delta_angle > 180:
    #             delta_angle -= 360

    #         if abs(delta_angle) <= threshold:
    #             botcontroller.send_control_command(0.0, 0.0)  # Stop steering

    #             print(f"Calibration complete: Delta angle {delta_angle}° within ±{threshold}°")
    #             return True

    #         if delta_angle < 0: # turn left
    #             steering *= -1 
    #         else:
    #             steering *= 1

    #         botcontroller.send_control_command(0.0, steering)  # High left tur
    #         time.sleep(1)
    #     # Timeout reached
    #     print(f"Calibration timeout after {duration}s. Final delta angle: {delta_angle}°")
    #     botcontroller.send_control_command(0.0, 0.0)  # Stop on timeout
    #     return False 

    def adjust_route_with_edge_snapping(self):
        '''If robot is between x and y nodes the robot goes to the nearest edge'''
        
        current_utm= self.gps_to_utm(self.start_gps[0],self.start_gps[1])
        current_utm = Point(current_utm[0], current_utm[1])
        u, v, _ = ox.distance.nearest_edges(self.graph_proj, current_utm.x, current_utm.y)
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
    def store_route_edges(self):
        """Extracts and stores the edge geometries for each segment in the route.
        Assumes all consecutive nodes are connected in the graph."""
        self.route_edges = []
        count = 0 
        for i in range(len(self.route) - 1):
            u = self.route[i]
            v = self.route[i + 1]

            try:
                edge_data = self.graph[u][v][0]
            except KeyError:
                edge_data = self.graph[v][u][0]  # If direction is reversed

            # Handle multiple edges between nodes
            if isinstance(edge_data, pd.DataFrame):
                geometry = edge_data.iloc[0]['geometry']
            else:
                geometry = edge_data['geometry']

            self.route_edges.append(geometry)
        # print("the length of the node is ", len(self.route_edges))
        current_utm= self.gps_to_utm(self.start_gps[0],self.start_gps[1])
        current_utm = Point(current_utm[0], current_utm[1])
        u, v, key = ox.distance.nearest_edges(self.graph_proj, current_utm.x, current_utm.y)
        if u in self.route and v in self.route:
            count +=2
        if count !=2 and (u in self.route or v in self.route):
 # If direction is reversed
            geometry = self.edges.loc[(u, v, key), 'geometry'] 
        self.route_edges.append(geometry)
        if self.route_length ==1 :
            destination_utm = self.gps_to_utm(self.destination_gps[0], self.destination_gps[1])
            destination_utm = Point(destination_utm[0],destination_utm[1])
            u, v, key = ox.distance.nearest_edges(self.graph_proj, destination_utm.x, destination_utm.y)
            if u in self.route and v in self.route:
                count +=2
            if count !=2 and (u in self.route or v in self.route):
    # If direction is reversed
                geometry = self.edges.loc[(u, v, key), 'geometry'] 
            self.route_edges.append(geometry)


    def _draw_static_map(self):
        """Draws the static GPS map and returns the figure and axes."""
        self.store_route_edges()

        # Create the base map with the graph
        fig, ax = ox.plot_graph(self.graph_proj, node_size=10, edge_color='gray', edge_linewidth=1, show=False, close=False)

        # Plot origin and destination points
        ax.scatter([self.origin.geometry.x.iloc[0]], [self.origin.geometry.y.iloc[0]], c='blue', s=100, label='Start')
        ax.scatter([self.destination.geometry.x.iloc[0]], [self.destination.geometry.y.iloc[0]], c='green', s=100, label='Goal')

        # Plot the route nodes as red points
        route_nodes = self.nodes.loc[self.route]
        ax.plot(route_nodes.geometry.x, route_nodes.geometry.y, color='red', linewidth=2, marker='o', markersize=5, label='Route Nodes')

        # Plot the route path (edges connecting the nodes)
        for i in range(len(self.route) - 1):
            node1 = self.route[i]
            node2 = self.route[i + 1]
            x1, y1 = self.nodes.loc[node1].geometry.x, self.nodes.loc[node1].geometry.y
            x2, y2 = self.nodes.loc[node2].geometry.x, self.nodes.loc[node2].geometry.y
            ax.plot([x1, x2], [y1, y2], color='orange', linewidth=2)  # Plotting the line between consecutive nodes

        plt.axis("off")  # Hide the axis for a cleaner map
        return fig, ax

    def draw_map(self):
        """Draws the static GPS map with all nodes, route, and route edges."""
        fig, ax = self._draw_static_map()  # Call the existing private method that returns fig and ax
        fig.savefig('/home/robotixx/Sameep/frodobot_workspace/src/localization/src/robot_navigation/static_map.png')

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
    
    