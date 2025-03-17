#!/bin/bash

# Path to your checkpoint file
CHECKPOINT_PATH="/home/kintou/Work/Robotixx/er/checkpoin/vltseg_checkpoint_cityscapes_2.pth"



# Path to config file
CONFIG_PATH="/home/kintou/Work/Robotixx/er/configs/mask2former_evaclip_2xb8_1k_frozen_gta2cityscapes.py"

# Path to test image
TEST_IMAGE="/home/kintou/Work/Robotixx/er/Input/images/bb.png"

# Output directory
OUTPUT_DIR="/home/kintou/Work/Robotixx/er/Input/images/output/"

# Run inference

python inf.py $CONFIG_PATH $CHECKPOINT_PATH --image $TEST_IMAGE --output-dir $OUTPUT_DIR --cfg-options load_from=$CHECKPOINT_PATH