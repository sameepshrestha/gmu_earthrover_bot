# Obtained from: https://github.com/open-mmlab/mmsegmentation
# Modified to fix https://github.com/open-mmlab/mmsegmentation/issues/3666
# Including a copy of the class here seems simpler than forcing users to install my fork of mmsegmentation.
# --------------------------------------------------------
# # Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple
from mmseg.registry import MODELS
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmengine.model import BaseModule

try:
    from mmdet.models.dense_heads import \
        Mask2FormerHead as MMDET_Mask2FormerHead
except ModuleNotFoundError:
    MMDET_Mask2FormerHead = BaseModule

from mmengine.structures import InstanceData
from torch import Tensor


from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList


@MODELS.register_module()
class Mask2FormerHeadFixed(MMDET_Mask2FormerHead):
    def __init__(self,
                 in_channels: List[int],
                 strides: List[int],  # Add missing argument
                 feat_channels: int,
                 num_classes: int,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            num_classes=num_classes,
            **kwargs
        )
        
        # Add stride handling if needed
        self.strides = strides
        self.out_channels = kwargs["out_channels"]
        
        # Original initialization
        self.cls_embed = nn.Linear(feat_channels, num_classes + 1)
        self.align_corners = False
        self.ignore_index = 255

        #feat_channels = kwargs['feat_channels']
        # self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_Mask2FormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long()

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas
    
    # def visualize_segmentation_mask(self, seg_logits: torch.Tensor):
    #     """
    #     Visualize the segmentation mask with 19 different colors for 19 classes.

    #     Args:
    #         seg_logits (torch.Tensor): The segmentation logits of shape (batch_size, num_classes, height, width).
    #     """
    #     # Convert logits to class predictions
    #     seg_pred = seg_logits.argmax(dim=1) 
    #     save_path = "/home/kintou/Work/Robotixx/VLTS/images/output/Segmentation_Output.png" # Shape: (batch_size, height, width)

    #     # Convert tensor to numpy array
    #     seg_pred_np = seg_pred.squeeze().cpu().numpy() 
    #     print(np.unique(seg_pred_np)) # Remove batch dimension and convert to numpy

    #     # Define 19 distinct colors for the 19 classes
    #     class_colors = {
    #         0: [0, 0, 0],        # Class 0: Black (Background)
    #         1: [255, 0, 0],      # Class 1: Red
    #         2: [0, 255, 0],      # Class 2: Green
    #         3: [0, 0, 255],      # Class 3: Blue
    #         4: [255, 255, 0],    # Class 4: Yellow
    #         5: [255, 0, 255],    # Class 5: Magenta
    #         6: [0, 255, 255],    # Class 6: Cyan
    #         7: [128, 0, 0],      # Class 7: Dark Red
    #         8: [0, 128, 0],      # Class 8: Dark Green
    #         9: [0, 0, 128],      # Class 9: Dark Blue
    #         10: [128, 128, 0],   # Class 10: Olive
    #         11: [128, 0, 128],   # Class 11: Purple
    #         12: [0, 128, 128],   # Class 12: Teal
    #         13: [192, 192, 192], # Class 13: Silver
    #         14: [128, 128, 128], # Class 14: Gray
    #         15: [255, 165, 0],   # Class 15: Orange
    #         16: [255, 192, 203], # Class 16: Pink
    #         17: [165, 42, 42],   # Class 17: Brown
    #         18: [0, 255, 127],  # Class 18: Spring Green
    #     }

    #     # Create an RGB image from the class predictions
    #     height, width = seg_pred_np.shape
    #     rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    #     for class_idx, color in class_colors.items():
    #         rgb_image[seg_pred_np == class_idx] = color

    #     # Display the segmentation mask
    #     plt.imshow(rgb_image)
    #     plt.axis('off')  # Hide axes
    #     plt.title("Segmentation Mask")
    #     plt.show()

    #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    #     plt.close()  # Close the plot to free memory

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.

        """

        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)

        return seg_logits
