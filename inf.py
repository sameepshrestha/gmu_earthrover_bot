import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from mmengine.config import Config, DictAction

from mmseg.apis import init_model
from torchvision.io import read_image, ImageReadMode


from mmengine.registry import MODELS
import vltseg  
from mmseg.utils import register_all_modules
import torchvision.transforms as transforms
import torch.nn.functional as F
from mmseg.structures import SegDataSample
import cv2

register_all_modules(True)



class ImageSegmenter:
    def __init__(self, config_path, checkpoint_path, device, cfg_options):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.cfg = Config.fromfile(config_path)
        self.cfg.merge_from_dict(cfg_options)
        self.model = MODELS.build(self.cfg.model)

        self.model = init_model(self.cfg, checkpoint_path, device=self.device)
        
    def segment_image(self, image_input):
        """
        Returns:
            numpy.ndarray: Segmentation mask as numpy array
        """
        with torch.no_grad():
            # Handle different input types from pipeline
            if isinstance(image_input, np.ndarray):
                # Assuming HWC format from pipeline, convert to CHW
                image = torch.from_numpy(image_input).permute(2, 0, 1)
            elif isinstance(image_input, torch.Tensor):
                # If already a tensor, ensure it's in CHW format
                if image_input.dim() == 3 and image_input.shape[-1] not in [3, 4]:
                    # If HWC, permute to CHW
                    image = image_input.permute(2, 0, 1)
                else:
                    image = image_input
            else:
                raise ValueError("image_input must be a NumPy array or PyTorch tensor")

            # Handle 4-channel (RGBA) input by dropping alpha channel
            if image.shape[0] == 4:
                image = image[:3]  # Keep only RGB channels, shape: [3, H, W]
            elif image.shape[0] != 3:
                raise ValueError(f"Expected 3 or 4 channels, got {image.shape[0]}")
                

            image = image.to(torch.float32).to(self.device).unsqueeze(0)
            
            # Upscale to 1024x2048
            target_size = (1024, 2048)  
            image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)


            
            data_sample = SegDataSample()
            data = {'inputs': image, 'data_samples': [data_sample]}
            
            preprocessed_data = self.model.data_preprocessor(data, training=False)
            inputs = preprocessed_data['inputs']
            print("five")
            result = self.model.predict(inputs)
            pred_mask = result[0].seg_logits.data.argmax(dim=0).cpu().numpy()

            pred_mask = pred_mask.astype(np.uint8)

            # Resize to target size (512, 1024) using nearest-neighbor interpolation
            pred_mask_pil = Image.fromarray(pred_mask)
            pred_mask_resized = pred_mask_pil.resize(
                (1024, 512),  # (width, height) for PIL
                Image.NEAREST
            )
            pred_mask_resized = np.array(pred_mask_resized)  # Shape: [512, 1024]

            return pred_mask_resized

def segment_image(image_input_path, config_path, checkpoint_path, device, cfg_options):
 
    segmenter = ImageSegmenter(config_path, checkpoint_path, device, cfg_options)
    return segmenter.segment_image(image_input_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='EVA-02+Mask2Former Inference')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--image', help='Path to input image for inference')
    parser.add_argument('--output-dir', default='output', help='Directory to save results')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to run on')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override settings in config')
    parser.add_argument('--debug-weights', action='store_true',
                       help='Debug model weights loading')
    args = parser.parse_args()
    print("cfg_options: ", args.cfg_options)
    # Segment the image
    pred_mask = segment_image(args.image, args.config, args.checkpoint, args.device, args.cfg_options)
    
    # Visualize result
    plt.figure(figsize=(10, 10))
    plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=18)
    plt.axis('off')
    plt.title('Segmented Output')
    plt.show()