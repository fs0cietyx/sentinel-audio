import torch
import argparse
import onnx
import onnxruntime as ort
import os
import sys

def export_to_onnx(model_path, output_path, dummy_input_shape=(1, 2, 257, 3)):
    print(f"Loading PyTorch model from {model_path}...")
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.models.anc_network import DCCRNPlaceholder
            
    model = DCCRNPlaceholder()
    
    if os.path.exists(model_path):
        # strict=False safely loads the hardened weights while stripping PyTorch's QAT observer nodes
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        print("Model weights loaded successfully.")
    else:
        print("Warning: Model weights not found.")
        
    model.eval()

    # Using a 512-sample chunk (10-15ms latency) at 16kHz with n_fft=512 -> 3 frames
    dummy_input = torch.randn(*dummy_input_shape)

    print(f"Exporting model to ONNX format at {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True, 
        opset_version=13, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output']
    )
    print("Export complete.")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX.")
    parser.add_argument("--model", type=str, default="model.pth")
    parser.add_argument("--output", type=str, default="model.onnx")
    args = parser.parse_args()
    export_to_onnx(args.model, args.output)
