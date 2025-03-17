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
    @cached_property
    def session(self):
        return requests.Session() 
    def __init__(self, base_url):
        self.base_url = base_url
        self.bot_data = None


    def parse_image(self, camera="front"):
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
        try:
            response = self.session.get(f"{self.base_url}/data")
            response.raise_for_status()
            data = response.json()
            parsed_data = data
            return parsed_data  # Return the parsed data directly
        except requests.exceptions.RequestException as e:
            print(f"Error fetching bot data: {e}")
            return None

    def moving_average(self, window_size):
        gps_lats = []
        gps_lots = []
        gps_orients = []
        prev_time = 0.0 
        while len(gps_orients) < window_size:
            data = self.fetch_bot_data()
            # print(data)
            if data is None:
                continue  # Skip this cycle if data wasn't fetched
            # Check for duplicate timestamps (if necessary)
            if  prev_time != data["timestamp"]:
                prev_time = data["timestamp"]
                gps_lots.append(data["longitude"])
                gps_lats.append(data["latitude"])
                gps_orients.append(data["orientation"])
                print(len(gps_lats))
            time.sleep(.25)  # Assuming a 1Hz update rate

        avg_latitude = sum(gps_lats) / window_size
        avg_longitude = sum(gps_lots) / window_size
        avg_heading = sum(gps_orients) / window_size
        return (avg_latitude, avg_longitude), avg_heading
