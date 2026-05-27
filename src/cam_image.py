import numpy as np
from matplotlib import pyplot as plt
import os
import hashlib
from src.helpers import conversation_save


class CAM_IMAGE:
    """Class Activation Mapping (CAM) for visualizing which parts of input images
    activate the network most strongly for specific predictions.
    
    Args:
        images (np.ndarray): Images to compute CAM for. Shape: (num_images, height, width, channels)
    """
    
    def __init__(self, images: np.ndarray):
        """Initialize CAM tracker with images.
        
        Args:
            images (np.ndarray): Images to compute CAM for. Shape: (num_images, height, width, channels)
        """
        assert images.ndim == 4, "Images must be 4D array (num_images, height, width, channels)"
        
        self.images = images
        self.num_images = images.shape[0]
        
        # Create unique hash for each image to track them across shuffled batches
        # This way, even if batch order changes, we identify the same image
        self.image_hashes = [self._hash_image(img) for img in images]
        
        # Store outputs and gradients keyed by image hash
        self.saved_outputs = []  # Activations from each layer
        self.saved_gradients = {h: [] for h in self.image_hashes}  # Gradients per image
        
        self.cam_maps = None  # Will store computed CAM heatmaps
    
    def _hash_image(image: np.ndarray) -> str:
        """Create a unique hash for an image to track it across batches.
        
        Uses SHA256 of image data so same image is always identified, even if shuffled.
        
        Args:
            image (np.ndarray): Image array. Shape: (height, width, channels)
        
        Returns:
            str: Hash string
        """
        return hashlib.sha256(image.tobytes()).hexdigest()
    
    def start_recording(self):
        """Reset recording buffers before a forward/backward pass."""
        self.saved_outputs = []
    
    def record_output(self, output: np.ndarray):
        """Record layer output during forward pass.
        
        Args:
            output (np.ndarray): Output from a layer. Shape: (batch_size, ...)
        """
        self.saved_outputs.append(output.copy())
    
    def record_gradient(self, batch_images: np.ndarray, delta: np.ndarray):
        """Record gradients during backward pass by matching batch images to CAM images.
        
        Automatically identifies which images in the batch are CAM images,
        regardless of batch order (handles shuffling).
        
        Args:
            batch_images (np.ndarray): Images from current batch. Shape: (batch_size, height, width, channels)
            delta (np.ndarray): Gradient tensor. Shape: (batch_size, ...)
        """
        # Find which positions in batch match our CAM images
        for batch_idx, batch_img in enumerate(batch_images):
            batch_hash = self._hash_image(batch_img)
            
            # If this batch image is one of our CAM images, record its gradient
            if batch_hash in self.saved_gradients:
                self.saved_gradients[batch_hash].append(delta[batch_idx:batch_idx+1].copy())
    
    def compute_cam(self, layer_idx: int = -1):
        """Compute Class Activation Maps using gradients from a specific layer.
        
        Uses: CAM = sum_k (grad_k * activation_k) where k is channel
        
        Args:
            layer_idx (int): Which layer to compute CAM from (default: -1 = last layer)
                           Must have saved outputs and gradients
        
        Returns:
            np.ndarray: CAM heatmaps. Shape: (num_images, height, width)
        """
        if layer_idx == -1:
            layer_idx = len(self.saved_outputs) - 1
        
        assert layer_idx < len(self.saved_outputs), f"Layer {layer_idx} not in saved outputs"
        
        activations = self.saved_outputs[layer_idx]  # (batch, h, w, c)
        
        self.cam_maps = []
        
        for img_idx, img_hash in enumerate(self.image_hashes):
            # Get gradients for this specific image
            if self.saved_gradients[img_hash]:
                # Average gradients if multiple passes were recorded
                avg_gradient = np.mean(
                    [g[0] if g.ndim > 2 else g for g in self.saved_gradients[img_hash]],
                    axis=0
                )
                
                # Get corresponding activation (use first batch element as reference shape)
                activation = activations[0]  # All batch elements have same spatial dims
                
                # Compute CAM: weighted sum of activations
                # avg_gradient shape: (h, w, c) or (c,) depending on layer
                if avg_gradient.ndim == 3:  # Conv layer: (h, w, c)
                    cam = np.sum(avg_gradient * activation, axis=2)
                elif avg_gradient.ndim == 1:  # Dense layer: (c,)
                    # Fallback for dense layers (less meaningful CAM)
                    cam = avg_gradient[0] if avg_gradient.shape[0] > 0 else np.zeros_like(activation[:, :, 0])
                else:
                    raise ValueError(f"Unexpected gradient shape: {avg_gradient.shape}")
                
                # Normalize CAM to [0, 1]
                cam = np.maximum(cam, 0)  # ReLU
                if cam.max() > 0:
                    cam = cam / cam.max()
                
                self.cam_maps.append(cam)
            else:
                # No gradients recorded for this image
                print(f"Warning: No gradients recorded for image {img_idx}. Was it in training batches?")
                self.cam_maps.append(np.zeros((activations.shape[1], activations.shape[2])))
        
        return np.array(self.cam_maps)
    
    def visualize_cam(self, original_images: np.ndarray = None, save_to: str = "", 
                      title: str = "CAM Visualization", show: bool = False, 
                      cols: int = 4, alpha: float = 0.5):
        """Visualize CAM heatmaps overlaid on original images.
        
        Args:
            original_images (np.ndarray): Original input images to overlay CAM on.
                                         Shape: (num_images, height, width, channels)
                                         If None, use self.images
            save_to (str): Folder path to save visualizations
            title (str): Title for saved plot
            show (bool): Whether to display plot
            cols (int): Number of columns in subplot grid
            alpha (float): Transparency of CAM overlay (0-1)
        """
        if self.cam_maps is None:
            raise ValueError("Must call compute_cam() before visualizing")
        
        if original_images is None:
            original_images = self.images
        
        num_images = len(self.cam_maps)
        rows = int(np.ceil(num_images / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.atleast_1d(axes).flatten()
        
        for i in range(rows * cols):
            ax = axes[i]
            ax.axis("off")
            
            if i < num_images:
                # Get original image
                orig_img = original_images[i]
                if orig_img.shape[-1] == 1:
                    orig_img = orig_img.squeeze(-1)
                
                # Display original
                if orig_img.ndim == 2:
                    ax.imshow(orig_img, cmap="gray")
                else:
                    ax.imshow(orig_img)
                
                # Overlay CAM heatmap
                cam_map = self.cam_maps[i]
                # Resize CAM to match image if needed
                if cam_map.shape != orig_img.shape[:2]:
                    from scipy.ndimage import zoom
                    scale = np.array(orig_img.shape[:2]) / np.array(cam_map.shape)
                    cam_map = zoom(cam_map, scale, order=1)
                
                ax.imshow(cam_map, cmap="hot", alpha=alpha)
                ax.set_title(f"Image {i}")
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        if save_to:
            def save_plot():
                os.makedirs(os.path.join("outputs", "images", save_to), exist_ok=True)
                plot_path = os.path.join("outputs", "images", save_to, f"{title}.png")
                plt.savefig(plot_path, dpi=100, bbox_inches="tight")
                print(f"CAM visualization saved to {plot_path}")
                plt.close()
            
            conversation_save(save_plot, save_to, minus_y=False)
        else:
            plt.close()
    
    def get_images(self) -> np.ndarray:
        """Get the CAM images.
        
        Returns:
            np.ndarray: Images. Shape: (num_images, height, width, channels)
        """
        return self.images
    
    def get_num_images(self) -> int:
        """Get number of CAM images.
        
        Returns:
            int: Number of images
        """
        return self.num_images
    
    def get_gradient_count(self):
        """Get how many times gradients were recorded for each image.
        
        Useful for debugging to verify CAM images appeared in batches.
        
        Returns:
            dict: Mapping of image_hash -> count
        """
        return {h: len(grads) for h, grads in self.saved_gradients.items()}