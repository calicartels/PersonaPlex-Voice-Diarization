import os
os.environ["NO_TORCH_COMPILE"] = "1"

import sys
from pathlib import Path
import torch
import torch.nn as nn
from safetensors.torch import load_file
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "moshi"))
from moshi.models.loaders import get_mimi

def component_info(component, name):
    total_params = sum(p.numel() for p in component.parameters())
    trainable_params = sum(p.numel() for p in component.parameters() if p.requires_grad)
    info = {'name': name, 'total_parameters': total_params, 'trainable_parameters': trainable_params, 'layers': []}
    info['dimension'] = getattr(component, 'dimension', None)
    info['num_layers'] = getattr(component, 'num_layers', None)
    info['transformer_layers'] = len(getattr(component, 'transformer', type('obj', (object,), {'layers': []})()).layers) if hasattr(component, 'transformer') else None
    return info

def structure(mimi):
    architecture = {'components': {}}
    architecture['components']['encoder'] = component_info(mimi.encoder, "Encoder")
    architecture['components']['encoder_transformer'] = component_info(mimi.encoder_transformer, "Encoder Transformer")
    architecture['components']['decoder'] = component_info(mimi.decoder, "Decoder")
    architecture['components']['decoder_transformer'] = component_info(mimi.decoder_transformer, "Decoder Transformer")
    architecture['components']['quantizer'] = component_info(mimi.quantizer, "Quantizer")
    return architecture

def activation_shapes(mimi):
    dummy_audio = torch.randn(1, 1, 24000)
    shapes = {'input': list(dummy_audio.shape)}
    
    with torch.no_grad():
        encoder_out = mimi.encoder(dummy_audio)
        shapes['encoder'] = list(encoder_out.shape)
        
        if mimi.encoder_transformer is not None:
            features = encoder_out.transpose(1, 2)
            features = mimi.encoder_transformer.input_proj(features) if mimi.encoder_transformer.input_proj is not None else features
            transformer_out = features
            for layer in mimi.encoder_transformer.transformer.layers:
                transformer_out = layer(transformer_out)
            transformer_out = transformer_out.transpose(1, 2)
            shapes['transformer'] = list(transformer_out.shape)
        
        latent = mimi._encode_to_unquantized_latent(dummy_audio)
        shapes['latent'] = list(latent.shape)
        
        codes = mimi.encode(dummy_audio)
        shapes['codes'] = list(codes.shape)
    
    return shapes

def weights(weights_path):
    state_dict = load_file(str(weights_path))
    return {
        'total_keys': len(state_dict),
        'encoder_keys': len([k for k in state_dict.keys() if k.startswith('encoder.')]),
        'transformer_keys': len([k for k in state_dict.keys() if 'transformer' in k]),
        'quantizer_keys': len([k for k in state_dict.keys() if 'quantizer' in k]),
        'decoder_keys': len([k for k in state_dict.keys() if k.startswith('decoder.')]),
    }

def save_results(architecture, output_dir):
    json_path = output_dir / "architecture.json"
    with open(json_path, 'w') as f:
        json.dump(architecture, f, indent=2)
    
    txt_path = output_dir / "architecture.txt"
    with open(txt_path, 'w') as f:
        f.write("COMPONENTS:\n")
        for comp_name, comp_info in architecture['components'].items():
            f.write(f"\n{comp_info['name']}:\n")
            f.write(f"  Parameters: {comp_info['total_parameters']:,}\n")
            f.write(f"  Dimension: {comp_info['dimension']}\n")
            f.write(f"  Transformer layers: {comp_info['transformer_layers']}\n")
        
        f.write("\n\nACTIVATION SHAPES:\n")
        for stage, shape in architecture['activation_shapes'].items():
            f.write(f"  {stage}: {shape}\n")
        
        f.write("\n\nWEIGHTS:\n")
        for key, value in architecture['weights'].items():
            f.write(f"  {key}: {value}\n")

def run_analysis(weights_path):
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    mimi = get_mimi(str(weights_path), device='cpu')
    mimi.eval()
    
    architecture = structure(mimi)
    architecture['activation_shapes'] = activation_shapes(mimi)
    architecture['weights'] = weights(weights_path)
    
    save_results(architecture, output_dir)

def main():
    weights_path = Path(__file__).parent.parent.parent / "weights" / "tokenizer-e351c8d8-checkpoint125.safetensors"
    run_analysis(weights_path)

if __name__ == "__main__":
    main()
