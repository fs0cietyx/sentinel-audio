import torch
import argparse
import onnx
import onnxruntime as ort

def export_to_onnx(model_path, output_path, dummy_input_shape=(1, 1, 16000)):
    print(f"Loading PyTorch model from {model_path}...")
    
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.models.anc_network import DCCRNPlaceholder
            
    model = DCCRNPlaceholder()
    
    # Prepare model for QAT so the architecture matches the saved weights
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    torch.ao.quantization.prepare_qat(model, inplace=True)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("Model weights loaded successfully.")
    else:
        print("Warning: Model weights not found, exporting with random initialization.")
        
    model.eval()
    torch.ao.quantization.convert(model, inplace=True)

    # The real model expects a complex spectrogram: (Batch, 2, Freq, Time)
    # Using a 512-sample chunk (10-15ms latency) at 16kHz with n_fft=512 -> 3 frames
    dummy_input_shape = (1, 2, 257, 3) 
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
        # dynamic_axes is explicitly removed. FPGAs require static memory allocation.
    )
    print("Export complete.")

    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX.")
    parser.add_argument("--model", type=str, default="model.pth", help="Path to PyTorch model weights.")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output path for ONNX model.")
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.output)
