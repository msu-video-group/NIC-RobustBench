import torch
import torch.nn.functional as F

def median_blur(images, kernel_size):
    """
    Applies median blur to a batch of images using PyTorch.
    
    Args:
        images (torch.Tensor): Input batch of images with shape (B, C, H, W).
        kernel_size (int): Size of the median blur kernel (must be odd).
        
    Returns:
        torch.Tensor: Blurred batch of images with shape (B, C, H, W).
    """
    # Pad the images
    pad_size = kernel_size // 2
    padded_images = F.pad(images, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    # Create sliding window view of the padded images
    window_size = (kernel_size, kernel_size)
    windows = padded_images.unfold(2, window_size[0], 1).unfold(3, window_size[1], 1)
    
    # Reshape the windows to (B, C, H*W, kernel_size*kernel_size)
    windows = windows.reshape(images.shape[0], images.shape[1], -1, kernel_size*kernel_size)
    
    # Apply median operation along the last dimension
    median_values, _ = torch.median(windows, dim=-1)
    
    # Reshape the median values back to the original image shape
    blurred_images = median_values.reshape(images.shape)
    
    return blurred_images

class Defense:
    def __init__(self, kernel_size=3):
        self.defence_name = 'median-blur'
        self.kernel_size = kernel_size

    def __call__(self, image):
        return median_blur(image, self.kernel_size)
   
