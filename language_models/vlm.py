import base64
import requests
import json
import os, time
import matplotlib as plt
import cv2
import ast
import numpy as np
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types
import re

# OpenAI API Key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
DIRECTIONS = ["front left", "turn left", "front right", "turn right", "front"]

def parse_text(text):
    strings = re.findall(r"\[(.*?)\]", text)
    return strings[0]


class VLM:
    def __init__(self):
        self.prompt = "The target is at {} {}. \
        The image is from the front camera of the robot. The robot prefers to run on flat areas, such as sidewalks and pavements.\
        Find a direction that robot can move in the next time step to guide the robot to the flat areas and in future to the goal. \
        The robot has these directions to choose: [{}] \
        Output the format: [direction], reason."

    def _encode_image(self, img):
        _, img_encoded = cv2.imencode('.jpeg', img)
        return base64.b64encode(img_encoded).decode('utf-8')

    def _parse_answer(self, result_string):
        start_index = result_string.find('[') + 1
        end_index = result_string.find(']')

        if end_index == -1:
            return -1
        # Get the number as a string
        return result_string[start_index:end_index]

    def predict(self, img, target=("front", "left")):
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[self.prompt.format(target[0], target[1], DIRECTIONS), img])
        print(response.text)
        return parse_text(response.text)

if __name__ == "__main__":
    vlm = VLM()
    image = Image.open("/home/jing/Downloads/erc/Apr21_mini_grass/FrontCamera/1745264711.353551.png")
    output = vlm.predict(img=image, target=("front", "right"))
    print("test")