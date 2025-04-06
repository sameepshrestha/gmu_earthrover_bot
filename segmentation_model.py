
# import numpy as np 
# import cv2
# import matplotlib.pyplot as plt 
# from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, AutoImageProcessor
# from PIL import Image
# import torch 
# class Segmentation_models:
#     def __init__(self):
#         self.feature_extractor = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
#         self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
#     def predict(self, image):
#         image_resized = image.resize((1024, 1024))
#         self.model.eval()
#         with torch.no_grad():
#             inputs = self.feature_extractor(images=image_resized, return_tensors="pt")
#             outputs = self.model(**inputs)
#         logits = outputs.logits
#         seg_mask = logits.argmax(dim=1).squeeze().cpu().numpy()
#         seg_mask = seg_mask.astype(np.uint8)
#         seg_mask_resized = Image.fromarray(seg_mask).resize(image.size, Image.NEAREST)
#         seg_mask_resized = np.array(seg_mask_resized)
#         return seg_mask_resized
    
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, AutoImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

new_transform = transforms.Compose([
    transforms.Resize((512, 1024)),  # Ensure consistent size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Segmentation_model:
    def __init__(self):
        """
        Initializes the segmentation model and feature extractor with CUDA support if available.
        """
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and move to GPU if available
        self.model = SegformerForSemanticSegmentation.from_pretrained("segments-tobias/segformer-b0-finetuned-segments-sidewalk")
        self.model = self.model.to(self.device)
        self.feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")


    def predict(self, image_input):

        image_input = np.array(image_input)
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        image = np.array(image_rgb)
        
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        self.model.eval()
        
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model(**inputs)
        
        logits = outputs.logits.cpu()
        predicted_labels = torch.argmax(logits, dim=1).squeeze(0).numpy()
        
        segmentation_resized = Image.fromarray(predicted_labels.astype(np.uint8)).resize(image.shape[1::-1], resample=Image.NEAREST)
        segmentation_resized = np.array(segmentation_resized)
        
        return segmentation_resized
    





class Segmentation_modelXL:
    def __init__(self):
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.feature_extractor = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        
        # Move the model to the appropriate device
        self.model.to(self.device)
        
    def predict(self, image):
        image_resized = image.resize((1024, 1024), Image.NEAREST)  # Use Image.NEAREST for no interpolation
        self.model.eval()
        with torch.no_grad():
            inputs = self.feature_extractor(images=image_resized, return_tensors="pt")
            
            # Move inputs to the appropriate device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        seg_mask = logits.argmax(dim=1).squeeze().cpu().numpy()  # Move logits back to CPU for numpy operations
        seg_mask = seg_mask.astype(np.uint8)
        seg_mask1 = np.array(seg_mask, copy=True)
        seg_mask_resized = Image.fromarray(seg_mask1).resize(image.size, Image.NEAREST)
        seg_mask_resized = np.array(seg_mask_resized)
        return seg_mask_resized

import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import numpy as np
from PIL import Image

class Mask2FormerSegmentor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
        
        self.model.to(self.device)
        
    def predict(self, image):
        image_resized = image.resize((512, 512), Image.NEAREST)  # Use NEAREST to avoid interpolation
        
        self.model.eval()
        with torch.no_grad():
            inputs = self.processor(images=image_resized, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.model(**inputs)
            seg_mask = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[(512, 512)])[0]
            seg_mask = seg_mask.cpu().numpy()  # Move to CPU and convert to NumPy
        seg_mask = seg_mask.astype(np.uint8)
        seg_mask_resized = Image.fromarray(seg_mask).resize(image.size, Image.NEAREST)
        seg_mask_resized = np.array(seg_mask_resized)
        return seg_mask_resized