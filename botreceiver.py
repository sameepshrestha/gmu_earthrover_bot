import os
import threading
import base64
from io import BytesIO
from PIL import Image
import requests
import time 
import cv2
from functools import cached_property



class BotReceiver:
    '''This class requests data and images from the server'''
    @cached_property
    def session(self):
        return requests.Session() 
    def __init__(self, base_url):
        self.base_url = base_url
        self.bot_data = None


    def parse_image(self, camera="front"):
        '''Parse the image from API'''
        try:
            response = self.session.get(f"{self.base_url}/v2/{camera}")
            response = response.json()
            frame = camera + "_frame"
            img_data = base64.b64decode(response[frame])
            img = Image.open(BytesIO(img_data))
            if img.mode == "RGBA":
                img = img.convert("RGB")
            return img
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    
    def fetch_bot_data(self):
        '''Parse the image from API'''
        try:
            response = self.session.get(f"{self.base_url}/data")
            response.raise_for_status()
            data = response.json()
            parsed_data = data
            return parsed_data  # Return the parsed data directly
        except requests.exceptions.RequestException as e:
            print(f"Error fetching bot data: {e}")
            return None
        
    def get_checkpoint(self):
        '''Get the checkpoint from the API'''
        try:
            destination = []
            response = self.session.get(f"{self.base_url}/checkpoints_list")
            response.raise_for_status()
            data = response.json()
            data = data["checkpoints_list"]
            for i in range(len(data)):
                destination.append(float(data[i]["latitude"]),float(data[i]["longitude"]))
            return destination
        except requests.exceptions.RequestException as e:
            print(f"Error fetching checkpoint data: {e}")
            return None
        

    def moving_average(self, window_size):
        '''Get Moving average of GPS at the start'''
        gps_lats = []
        gps_lots = []
        gps_orients = []
        prev_time = 0.0 
        while len(gps_orients) < window_size:
            data = self.fetch_bot_data()
            print(data)
            # print(data)
            if data is None:
                continue  # Skip this cycle if data wasn't fetched
            # Check for duplicate timestamps (if necessary)
            if  prev_time != data["timestamp"]:
                prev_time = data["timestamp"]
                gps_lots.append(data["longitude"])
                gps_lats.append(data["latitude"])
                gps_orients.append(data["orientation"])
                print(data["latitude"],data["longitude"],data["orientation"])
            time.sleep(.25)  # Assuming a 1Hz update rate

        avg_latitude = sum(gps_lats) / window_size
        avg_longitude = sum(gps_lots) / window_size
        avg_heading = sum(gps_orients) / window_size
        return (avg_latitude, avg_longitude), avg_heading
    
#     'battery': 30, 'signal_level': 5, 'orientation': 234, 'lamp': 1852793600, 'speed': 0, 'gps_signal': 33.5, 'latitude': 38.82761001586914, 'longitude': -77.30582427978516, 'vibration': 0.69, 'accels': [[1.001, 0.013, 0.028, 1743736257.879], [1.003, 0.004, 0.027, 1743736257.985], [0.999, -0.015, 0.019, 1743736258.094], [1.003, 0.017, 0.018, 1743736258.206], [1.003, 0.008, 0.026, 1743736258.318]], 'gyros': [[-0.138, -0.065, 0.326, 1743736257.979], [-0.352, -0.248, 0.547, 1743736258.088], [-0.206, -0.386, 0.517, 1743736258.192], [-0.1, -0.294, 0.311, 1743736258.302], [-0.153, 0.042, 0.654, 1743736258.402]], 'mags': [[-1215, -710, 508, 1743736258.259]], 'rpms': [[0, 0, 0, 0, 1743736201.597], [0, 0, 0, 0, 1743736201.611], [0, 0, 0, 0, 1743736232.082], [0, 0, 0, 0, 1743736232.093], [0, 0, 0, 0, 1743736232.119]], 'timestamp': '1743736258.564000'}
# 38.82761001586914 -77.30582427978516 234
# {'battery': 69, 'signal_level': 5, 'orientation': 107, 'lamp': 1852793600, 'speed': 0, 'gps_signal': 37.5, 'latitude': 30.482446670532227, 'longitude': 114.30267333984375, 'vibration': 0.32, 'accels': [[1.002, -0.006, -0.01, 1743736258.245], [1.004, -0.004, -0.01, 1743736258.35], [1.001, -0.005, -0.009, 1743736258.451], [0.999, -0.006, -0.008, 1743736258.564], [1.001, -0.005, -0.01, 1743736258.665]], 'gyros': [[0.364, -0.35, -0.886, 1743736258.207], [0.387, -0.472, -0.795, 1743736258.313], [0.41, -0.334, -0.871, 1743736258.42], [0.402, -0.411, -0.833, 1743736258.522], [0.471, -0.357, -0.863, 1743736258.632]], 'mags': [[-895, 340, 102, 1743736258.389]], 'rpms': [[0, 0, 0, 0, 1743736209.296], [0, 0, 0, 0, 1743736209.322], [0, 0, 0, 0, 1743736239.794], [0, 0, 0, 0, 1743736239.804], [0, 0, 0, 0, 1743736239.829]], 'timestamp': '1743736258.894000'}
# 30.482446670532227 114.30267333984375 107
# {'battery': 30, 'signal_level': 5, 'orientation': 234, 'lamp': 1852793600, 'speed': 0, 'gps_signal': 33.5, 'latitude': 38.82761001586914, 'longitude': -77.30582427978516, 'vibration': 0.76, 'accels': [[1.003, -0.008, 0.035, 1743736258.431], [1.002, 0.008, 0.032, 1743736258.544], [1.001, 0.009, 0.027, 1743736258.644], [1.003, 0.005, 0.016, 1743736258.747], [1.001, 0.003, 0.026, 1743736258.86]], 'gyros': [[-0.153, 0.042, 0.654, 1743736258.402], [-0.046, 0.065, 0.41, 1743736258.503], [-0.245, -0.179, 0.532, 1743736258.611], [-0.214, -0.363, 0.441, 1743736258.716], [-0.199, -0.172, 0.418, 1743736258.813]], 'mags': [[-1225, -700, 563, 1743736258.763]], 'rpms': [[0, 0, 0, 0, 1743736201.597], [0, 0, 0, 0, 1743736201.611], [0, 0, 0, 0, 1743736232.082], [0, 0, 0, 0, 1743736232.093], [0, 0, 0, 0, 1743736232.119]], 'timestamp': '1743736259.030000'}
# 38.82761001586914 -77.30582427978516 234
# {'battery': 69, 'signal_level': 5, 'orientation': 108, 'lamp': 1852793600, 'speed': 0, 'gps_signal': 38, 'latitude': 30.482446670532227, 'longitude': 114.30267333984375, 'vibration': 0.25, 'accels': [[1.001, -0.005, -0.011, 1743736258.77], [1.001, -0.005, -0.016, 1743736258.874], [1.001, -0.006, -0.012, 1743736258.986], [1.003, -0.005, -0.006, 1743736259.096], [1.003, -0.005, -0.01, 1743736259.198]], 'gyros': [[0.402, -0.35, -0.787, 1743736258.74], [0.44, -0.365, -0.825, 1743736258.836], [0.425, -0.327, -0.863, 1743736258.947], [0.372, -0.411, -0.856, 1743736259.056], [0.379, -0.365, -0.787, 1743736259.169]], 'mags': [[-890, 325, 92, 1743736258.894]], 'rpms': [[0, 0, 0, 0, 1743736209.296], [0, 0, 0, 0, 1743736209.322], [0, 0, 0, 0, 1743736239.794], [0, 0, 0, 0, 1743736239.804], [0, 0, 0, 0, 1743736239.829]], 'timestamp': '1743736259.388000'}
# 30.482446670532227 114.30267333984375 108
# {'battery': 30, 'signal_level': 5, 'orientation': 233, 'lamp': 1852793600, 'speed': 0, 'gps_signal': 34, 'latitude': 38.82761001586914, 'longitude': -77.30582427978516, 'vibration': 0.63, 'accels': [[1.002, 0.012, 0.03, 1743736258.968], [1.001, 0, 0.029, 1743736259.076], [1.002, 0.007, 0.028, 1743736259.189], [1.004, 0.02, 0.033, 1743736259.306], [0.999, -0.004, 0.017, 1743736259.407]], 'gyros': [[-0.145, -0.103, 0.479, 1743736258.915], [-0.161, -0.179, 0.631, 1743736259.024], [-0.206, -0.073, 0.433, 1743736259.134], [-0.161, -0.225, 0.425, 1743736259.245], [-0.191, -0.256, 0.624, 1743736259.351]], 'mags': [[-1197, -712, 536, 1743736259.263]], 'rpms': [[0, 0, 0, 0, 1743736201.597], [0, 0, 0, 0, 1743736201.611], [0, 0, 0, 0, 1743736232.082], [0, 0, 0, 0, 1743736232.093], [0, 0, 0, 0, 1743736232.119]], 'timestamp': '1743736259.539000'}
# 38.82761001586914 -77.30582427978516 233
# {'battery': 30, 'signal_level': 5, 'orientation': 233, 'lamp': 1852793600, 'speed': 0, 'gps_signal': 34, 'latitude': 38.82761001586914, 'longitude': -77.30582427978516, 'vibration': 0.63, 'accels': [[1.002, 0.012, 0.03, 1743736258.968], [1.001, 0, 0.029, 1743736259.076], [1.002, 0.007, 0.028, 1743736259.189], [1.004, 0.02, 0.033, 1743736259.306], [0.999, -0.004, 0.017, 1743736259.407]], 'gyros': [[-0.145, -0.103, 0.479, 1743736258.915], [-0.161, -0.179, 0.631, 1743736259.024], [-0.206, -0.073, 0.433, 1743736259.134], [-0.161, -0.225, 0.425, 1743736259.245], [-0.191, -0.256, 0.624, 1743736259.351]], 'mags': [[-1197, -712, 536, 1743736259.263]], 'rpms': [[0, 0, 0, 0, 1743736201.597], [0, 0, 0, 0, 1743736201.611], [0, 0, 0, 0, 1743736232.082], [0, 0, 0, 0, 1743736232.093], [0, 0, 0, 0, 1743736232.119]], 'timestamp': '1743736259.539000'}
# {'battery': 30, 'signal_level': 5, 'orientation': 231, 'lamp': 1852793600, 'speed': 0, 'gps_signal': 34, 'latitude': 38.82761001586914, 'longitude': -77.30582427978516, 'vibration': 0.54, 'accels': [[0.999, -0.004, 0.017, 1743736259.407], [1, 0.002, 0.026, 1743736259.521], [1.002, 0.014, 0.034, 1743736259.634], [1.002, -0.002, 0.032, 1743736259.746], [1.001, -0.003, 0.027, 1743736259.858]], 'gyros': [[-0.123, 0.057, 0.364, 1743736259.447], [-0.138, -0.309, 0.494, 1743736259.548], [-0.268, -0.34, 0.532, 1743736259.657], [-0.184, -0.118, 0.509, 1743736259.756], [-0.13, -0.225, 0.586, 1743736259.864]], 'mags': [[-1197, -705, 563, 1743736259.78]], 'rpms': [[0, 0, 0, 0, 1743736201.597], [0, 0, 0, 0, 1743736201.611], [0, 0, 0, 0, 1743736232.082], [0, 0, 0, 0, 1743736232.093], [0, 0, 0, 0, 1743736232.119]], 'timestamp': '1743736260.041000'}
# 38.82761001586914 -77.30582427978516 231
# {'battery': 69, 'signal_level': 5, 'orientation': 107, 'lamp': 1852793600, 'speed': 0, 'gps_signal': 37, 'latitude': 30.482446670532227, 'longitude': 114.30267333984375, 'vibration': 0.22, 'accels': [[1.001, -0.005, -0.012, 1743736259.731], [0.999, -0.006, -0.009, 1743736259.849], [1.003, -0.005, -0.01, 1743736259.965], [1.003, -0.008, -0.014, 1743736260.066], [1.003, -0.005, -0.01, 1743736260.179]], 'gyros': [[0.379, -0.373, -0.818, 1743736259.788], [0.387, -0.334, -0.787, 1743736259.886], [0.364, -0.334, -0.81, 1743736259.996], [0.395, -0.388, -0.787, 1743736260.102], [0.372, -0.357, -0.848, 1743736260.205]], 'mags': [[-877, 332, 100, 1743736259.916]], 'rpms': [[0, 0, 0, 0, 1743736209.296], [0, 0, 0, 0, 1743736209.322], [0, 0, 0, 0, 1743736239.794], [0, 0, 0, 0, 1743736239.804], [0, 0, 0, 0, 1743736239.829]], 'timestamp': '1743736260.405000'}
# 30.482446670532227 114.30267333984375 107
