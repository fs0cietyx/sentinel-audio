#!/bin/bash
# -------------------------------------------------------------------
# Vitis AI ONNX to XModel Compilation Script for AMD PYNQ Boards
# 
# Run this inside the official Vitis AI Docker container:
# docker run -v $(pwd):/workspace -w /workspace -it xilinx/vitis-ai:latest
# -------------------------------------------------------------------

echo "Compiling fpga_model.onnx for PYNQ DPU..."

# We target the standard DPU architecture for Zynq/PYNQ (e.g. DPUCZDX8G)
# Change the arch.json to match your specific board (e.g. KV260, ZCU104)
TARGET_ARCH="/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json"

if [ ! -f "checkpoints/fpga_model.onnx" ]; then
    echo "Error: fpga_model.onnx not found. Run export_onnx.py first."
    exit 1
fi

vai_c_onnx \
    --model checkpoints/fpga_model.onnx \
    --arch $TARGET_ARCH \
    --output_dir checkpoints/ \
    --net_name dccrn_anc

echo "Compilation complete! The XModel (dccrn_anc.xmodel) is ready to be moved to the PYNQ board."
