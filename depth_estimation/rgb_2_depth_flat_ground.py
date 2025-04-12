#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from std_msgs.msg import Header

class DepthEstimatorNode:
    def __init__(self):
        rospy.init_node('flat_ground_depth_estimator')

        self.camera_height = 0.5  # in meters
        self.focal_length = 525.0  # in pixels 
        self.camera_tilt_deg = 0.0 # Angle in radians: downward tilt is positive
        self.bridge = CvBridge()

        rospy.Subscriber('/zed2i/zed_node/left_raw/image_raw_color/compressed',
                         CompressedImage, self.image_callback)

        self.depth_pub = rospy.Publisher('/depth_estimation', Image, queue_size=1)
        self.compressed_depth_pub = rospy.Publisher('/depth_estimation/compressed', CompressedImage, queue_size=1)


    def image_callback(self, msg):
        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        height, width = rgb_image.shape[:2]
        v0 = height / 2.0

        # Create depth map
        depth_map = np.zeros((height, width), dtype=np.float32)

        camera_tilt_rad = np.deg2rad(self.camera_tilt_deg)

        for v in range(height):
            pixel_angle = np.arctan2((v - v0), self.focal_length)  # ray angle from image center
            total_angle = camera_tilt_rad + pixel_angle

            if total_angle <= 0:  # Ray goes above the horizon
                depth = 10.0
            else:
                depth = self.camera_height / np.tan(total_angle)
            depth_map[v, :] = depth

            depth_map[v, :] = depth

        # Convert and publish depth image
        depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding="32FC1")
        depth_msg.header = Header()
        depth_msg.header.stamp = rospy.Time.now()
        self.depth_pub.publish(depth_msg)

        # Normalize depth map for visualization 
        depth_vis = np.nan_to_num(depth_map)
        depth_vis[np.isinf(depth_vis)] = 0.0
        depth_vis = np.clip(depth_vis, 0, 10)  # max 10 meters
        depth_vis = (depth_vis / 10.0 * 255).astype(np.uint8)


        # Encode to JPEG
        success, encoded_image = cv2.imencode('.jpg', depth_vis)
        if success:
            compressed_msg = CompressedImage()
            compressed_msg.header = depth_msg.header
            compressed_msg.format = "jpeg"
            compressed_msg.data = encoded_image.tobytes()
            self.compressed_depth_pub.publish(compressed_msg)


if __name__ == '__main__':
    try:
        DepthEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
