#!/bin/bash
set -e

echo "--- 1. Environment Check ---"
pip install lightning xformers zarr numpy torch --quiet

# Hardware Detection
if command -v nvidia-smi &> /dev/null; then
    ACCEL="gpu"
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    echo "Hardware: NVIDIA GPU detected (${VRAM}MB VRAM)"
else
    ACCEL="cpu"
    echo "Hardware: CPU mode (No GPU found)"
fi

echo "--- 2. Physics-Informed Data Generation ---"
python data_engine.py

echo "--- 3. Scalable Training Pipeline ---"
python main.py --accel $ACCEL

echo "--- 4. Production Inference Test ---"
# Quick test using the trained weights
python -c "import torch; from model_factory import ScalableGyroNet; m = ScalableGyroNet.load_from_checkpoint('models/gyro_final.ckpt'); print('System Ready for Production')"
