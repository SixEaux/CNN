"""ASCII representations of neural network layers for visualization."""

from src.dense import Dense
from src.convolution import Convolutional
from src.activation import Activation
from src.pooling import MaxPool
from src.flattening import Flattening
from src.dropout import Dropout
from src.loss import Loss


def convolutional(layer: Convolutional, index: int) -> str:
    kernels = str(layer.number_kernels)
    size = str(layer.size_kernel)
    stride = str(layer.stride)
    padding = str(layer.padding)
    out_dim = str(layer.out_dim) if layer.out_dim else 'N/A'
    
    ascii_art = f"""
    ╔════════════════════════════════════════╗
    ║  CONVOLUTIONAL LAYER {index}
    ╠════════════════════════════════════════╣
    ║  Kernels: {kernels}
    ║  Kernel Size: {size}x{size}
    ║  Stride: {stride}
    ║  Padding: {padding}
    ║  Output Shape: {out_dim}
    ╚════════════════════════════════════════╝
    """
    return ascii_art

def dense(layer: Dense, index: int) -> str:
    neurons = str(layer.number_neurons)
    init = str(layer.initialization)
    weight_shape = str(layer.weight.shape) if layer.weight is not None else 'N/A'
    
    ascii_art = f"""
    ╔════════════════════════════════════════╗
    ║  DENSE LAYER {index}
    ╠════════════════════════════════════════╣
    ║  Neurons: {neurons}
    ║  Initialization: {init}
    ║  Weight Shape: {weight_shape}
    ╚════════════════════════════════════════╝
    """
    return ascii_art

def activation(layer: Activation, index: int) -> str:
    func_name = str(layer.function)
    
    ascii_art = f"""
    ╔════════════════════════════════════════╗
    ║  ACTIVATION LAYER {index}
    ╠════════════════════════════════════════╣
    ║  Function: {func_name}
    ╚════════════════════════════════════════╝
    """
    return ascii_art

def pooling(layer: MaxPool, index: int) -> str:
    size = str(layer.size_kernel)
    stride = str(layer.stride)
    out_dim = str(layer.out_dim) if hasattr(layer, 'out_dim') and layer.out_dim else 'N/A'
    
    ascii_art = f"""
    ╔════════════════════════════════════════╗
    ║  MAX POOLING LAYER {index}
    ╠════════════════════════════════════════╣
    ║  Pool Size: {size}x{size}
    ║  Stride: {stride}
    ║  Output Shape: {out_dim}
    ╚════════════════════════════════════════╝
    """
    return ascii_art

def flattening(layer: Flattening, index: int) -> str:
    ascii_art = f"""
    ╔════════════════════════════════════════╗
    ║  FLATTENING LAYER {index}
    ╠════════════════════════════════════════╣
    ║  Reshapes 3D → 1D
    ║  Preserves total elements
    ╚════════════════════════════════════════╝
    """
    return ascii_art

def dropout(layer: Dropout, index: int) -> str:
    drop_rate = str(layer.drop_rate)
    
    ascii_art = f"""
    ╔════════════════════════════════════════╗
    ║  DROPOUT LAYER {index}
    ╠════════════════════════════════════════╣
    ║  Drop Rate: {drop_rate}
    ║  (Applied during training only)
    ╚════════════════════════════════════════╝
    """
    return ascii_art

def loss(layer: Loss, index: int = None) -> str:
    index_str = f" {index}" if index is not None else ""
    func = str(layer.function)
    classes = str(layer.nb_classes)
    
    ascii_art = f"""
    ╔════════════════════════════════════════╗
    ║  LOSS LAYER{index_str}
    ╠════════════════════════════════════════╣
    ║  Function: {func}
    ║  Classes: {classes}
    ╚════════════════════════════════════════╝
    """
    return ascii_art

def get_layer_ascii(layer, index: int) -> str:
    if isinstance(layer, Convolutional):
        return convolutional(layer, index)
    elif isinstance(layer, Dense):
        return dense(layer, index)
    elif isinstance(layer, Activation):
        return  activation(layer, index)
    elif isinstance(layer, MaxPool):
        return pooling(layer, index)
    elif isinstance(layer, Flattening):
        return flattening(layer, index)
    elif isinstance(layer, Dropout):
        return dropout(layer, index)
    elif isinstance(layer, Loss):
        return loss(layer, index)
    else:
        return f"\n╔══════════════════════════════════════╗\n║  UNKNOWN LAYER {index:<19} ║\n╚══════════════════════════════════════╝\n"


def print_network(layers: list, loss_layer: Loss = None):

    print("\n" + "=" * 40)
    print("          NETWORK ARCHITECTURE")
    print("=" * 40)

    for i, layer in enumerate(layers):
        print(get_layer_ascii(layer, i))
        if i < len(layers) - 1:
            print("                 ↓")

    if loss_layer is not None:
        print("                 ↓")
        print(loss(loss_layer))

    print("=" * 40 + "\n")


def save_network_ascii(layers: list, loss_layer: Loss = None, save_path: str = "outputs/architecture.txt"):

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("=" * 40 + "\n")
        f.write("          NETWORK ARCHITECTURE\n")
        f.write("=" * 40 + "\n\n")
        
        for i, layer in enumerate(layers):
            f.write(get_layer_ascii(layer, i))
            if i < len(layers) - 1:
                f.write("                 ↓\n")
        
        if loss_layer is not None:
            f.write("                 ↓\n")
            f.write(loss(loss_layer))
        
        f.write("\n" + "=" * 40)

    print(f"Network architecture saved to {save_path}")
