import numpy as np 
import cv2
import matplotlib.pyplot as plt
# class PerspectiveTransformer:
#     def __init__(self,  H=None):
#         self.h = H
#     def isolate_class(self, segmentation_whole):
#         binary_mask = np.zeros_like(segmentation_whole)
#         binary_mask[(segmentation_whole >= 0) & (segmentation_whole <= 1)] = 1
#         return binary_mask

#     def inverse_perspective_mapping(self, segmentation_whole):
#         segmentation_mask = self.isolate_class(segmentation_whole)
#         segmentation_mask = segmentation_mask.astype(np.uint8)  # Ensure the mask is uint8
#         # blurred_mask = cv2.GaussianBlur(segmentation_mask, (3, 3), 0)
#         # kernel = np.ones((3, 3), np.uint8)
#         # eroded_mask = cv2.erode(blurred_mask, kernel, iterations=1)
#         # contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         # if not contours:
#         #     raise ValueError("No contours found for the selected class in the segmentation mask.")
#         # largest_contour = max(contours, key=cv2.contourArea)
#         # contour_points = largest_contour[:, 0, :]  # Extract contour points
#         # topmost_y = np.min(contour_points[:, 1])
#         image_width = segmentation_whole.shape[1]
#         image_height = segmentation_whole.shape[0]
#         src_points = np.array([
#             [0, 300],                 # Top-left
#             [image_width, 300],       # Top-right
#             [image_width, image_height],    # Bottom-right
#             [0, image_height]               # Bottom-left
#         ], dtype=np.float32)
#         dst_points = np.array([
#             [0, 0],                        # Top-left
#             [image_width, 0],              # Top-right
#             [image_width, 200],   # B300m-right
#             [0, 200]              # Bottom-left
#         ], dtype=np.float32)
#         if self.h is None:
#             self.h = cv2.getPerspectiveTransform(src_points, dst_points)
#         # Warp the frame and segmentation mask to BEV
#         warped_segmentation = cv2.warpPerspective(segmentation_whole, self.h, (image_height,200), flags=cv2.INTER_NEAREST)
#         return  warped_segmentation


class PerspectiveTransformer:
    def __init__(self):
        self.left_source = np.float32([
        [200, 318],   # Top left
        [723, 318],   # Top right
        [1023, 576],  # Bottom right
        [0, 576]      # Bottom left
    ])

        self.left_dest = np.float32([
        [0, 0],      # Top-left
        [1023, 0],    # Top-right
        [1023, 720],  # Bottom-right
        [0, 720]     # Bottom-left
    ])
    
    def inverse_perspective_mapping(self,segmentation_whole):
        output_size = (1024,720)

        M_left = cv2.getPerspectiveTransform(self.left_source, self.left_dest)

        leftwarped_segementation = cv2.warpPerspective(segmentation_whole,M_left, output_size)
        # rightwarped_segmentation = cv2.warpPerspective(segmentation_whole,self.right_h,(540,720))
        # combined = np.hstack((leftwarped_segementation, rightwarped_segmentation))
        return leftwarped_segementation
