import time

import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from transformers import pipeline
from accelerate.test_utils.testing import get_backend
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
from PIL import Image
import requests
import matplotlib.pyplot as plt

class ModelTypes:
    depth_anything = "depth_anything"  # 0.26s
    intel_zoo = "intel_zoo"  # 0.4s

class DepthEstimation:
    def __init__(self, model_type):
        self.type = model_type
        if self.type == ModelTypes.depth_anything:
            checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
            self.pipe = pipeline("depth-estimation", model=checkpoint, device=device)
        elif self.type == ModelTypes.intel_zoo:
            checkpoint = "Intel/zoedepth-nyu-kitti"
            self.image_processor = AutoImageProcessor.from_pretrained(checkpoint)
            self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(device)

    def step(self, image):
        if self.type == ModelTypes.depth_anything:
            image = Image.fromarray(image)
            predictions = self.pipe(image)
            depth = predictions["depth"]
        elif self.type == ModelTypes.intel_zoo:
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                outputs = self.model(pixel_values)
            post_processed_output = self.image_processor.post_process_depth_estimation(
                outputs,
                source_sizes=[(image.shape[0], image.shape[1])],
            )
            predicted_depth = post_processed_output[0]["predicted_depth"]
            depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
            depth = depth.detach().cpu().numpy() * 255
        return depth

    def visualize(self, image, depth):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

        axes[0].imshow(image)
        axes[0].set_title("Image 1")
        axes[0].axis("off")

        axes[1].imshow(depth)
        axes[1].set_title("Image 2")
        axes[1].axis("off")

        plt.show()


if __name__ == "__main__":
    device, _, _ = get_backend()
    depth_estimator = DepthEstimation(model_type=ModelTypes.depth_anything)
    image1 = cv2.imread("/home/jing/Documents/erc/gmu_earthrover_bot/image_1740781714244.png")
    average_time = []
    for i in range(100):
        start_time = time.time()
        depth1 = depth_estimator.step(image1)
        average_time.append(time.time() - start_time)
        print("Step Time: {}; Average Time: {}".format(time.time() - start_time, np.mean(average_time)))
    # depth_estimator.visualize(image1, depth1)