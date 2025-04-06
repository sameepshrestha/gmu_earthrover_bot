import math
import os.path
import pickle
import time

import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from transformers import pipeline
from accelerate.test_utils.testing import get_backend
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
from PIL import Image
from tqdm import tqdm
import requests
import matplotlib.pyplot as plt


class ModelTypes:
    depth_anything = "depth_anything_v2"  # 0.26s
    intel_zoo = "intel_zoo"  # 0.4s


class DepthEstimation:
    def __init__(self, model_type, device):
        self.type = model_type
        self.device = device
        if self.type == ModelTypes.depth_anything:
            checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
            self.pipe = pipeline("depth-estimation", model=checkpoint, device=self.device)
        elif self.type == ModelTypes.intel_zoo:
            checkpoint = "Intel/zoedepth-nyu-kitti"
            self.image_processor = AutoImageProcessor.from_pretrained(checkpoint)
            self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(self.device)

    def step(self, image):
        if self.type == ModelTypes.depth_anything:
            image = Image.fromarray(image)
            predictions = self.pipe(image)
            depth = predictions["depth"]
        elif self.type == ModelTypes.intel_zoo:
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.to(self.device)
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

    def visualize(self, image, depth, estimated_depth):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 2 columns

        axes[0].imshow(image)
        axes[0].set_title("Image 1")
        axes[0].axis("off")

        axes[1].imshow(depth)
        axes[1].set_title("Image 2")
        axes[1].axis("off")

        axes[2].imshow(estimated_depth)
        axes[2].set_title("Image 3")
        axes[2].axis("off")
        plt.show()



def mae(depth1, depth2):
    return np.mean(np.abs(depth1 - depth2))


def test_single(depth_estimator, directory, depth_reference_val=10.0):
    rs_image = cv2.imread(os.path.join(directory, "rs_rgb.jpg"))
    rs_depth = np.load(os.path.join(directory, "rs_depth.npy"))
    average_time = []
    average_accuracy = []
    start_time = time.time()

    rs_image = cv2.resize(rs_image, [rs_depth.shape[1], rs_depth.shape[0]])

    depth1 = depth_estimator.step(rs_image)
    depth1 = np.asarray(depth1) * depth_reference_val / 255.0

    average_time.append(time.time() - start_time)
    average_accuracy.append(mae(rs_depth, depth1))
    print("Accuracy: {}, Average Time: {}".format(np.mean(average_accuracy), np.mean(average_time)))

    depth_estimator.visualize(rs_image, np.clip(rs_depth, a_min=0, a_max=depth_reference_val), depth1)


def test_folder(depth_estimator, directory, depth_reference_val=10):
    average_time = []
    average_accuracy = []
    for idx in tqdm(range(20)):
        start_time = time.time()
        with open(directory + "/{}.pkl".format(idx), "rb") as input_file:
            data = pickle.load(input_file)
        rs_image = data["camera"][-1]
        rs_depth = data["depth"][-1]
        rs_image = cv2.resize(rs_image, [rs_depth.shape[1], rs_depth.shape[0]])

        depth1 = depth_estimator.step(rs_image)
        depth1 = np.asarray(depth1) * depth_reference_val / 255.0

        average_time.append(time.time() - start_time)
        average_accuracy.append(mae(rs_depth, depth1))

        depth_estimator.visualize(rs_image, rs_depth, depth1)
    print("Accuracy: {}, Average Time: {}".format(np.mean(average_accuracy), np.mean(average_time)))


if __name__ == "__main__":
    device, _, _ = get_backend()
    depth_estimator = DepthEstimation(model_type=ModelTypes.depth_anything, device="cuda")

    Height = 42.0 / 100.0  # cm -> m
    FOV = 70 * math.pi / 180.0

    test_single(depth_estimator, depth_reference_val=10, directory="/home/jing/Documents/erc/gmu_earthrover_bot/depth_estimation/data")
    # test_folder(depth_estimator, depth_reference_val=10, directory="/home/jing/Documents/erc/gmu_earthrover_bot/depth_estimation/data/multiple_files")
    print("test")